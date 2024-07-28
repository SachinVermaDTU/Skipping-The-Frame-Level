import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch_optimizer as optim
import argparse
from .Model_ablation import TransKun
from .TrainUtil import initializeCheckpoint, load_checkpoint, save_checkpoint, average_gradients, MovingBuffer, doValidation

# Import your dataset and preprocessing functions
from .Data import DatasetMaestroIterator, collate_fn
from .preprocessing import preprocess_audio, preprocess_midi  # Ensure these functions are implemented

class AudioMidiDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, midi_original_dir, midi_produced_dir, transform=None):
        self.audio_dir = audio_dir
        self.midi_original_dir = midi_original_dir
        self.midi_produced_dir = midi_produced_dir
        self.transform = transform
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio_path = os.path.join(self.audio_dir, audio_file)
        midi_original_path = os.path.join(self.midi_original_dir, audio_file.replace('.mp3', '.mid'))
        midi_produced_path = os.path.join(self.midi_produced_dir, audio_file.replace('.mp3', '_produced.mid'))
        
        audio_data = preprocess_audio(audio_path)
        midi_original_data = preprocess_midi(midi_original_path)
        midi_produced_data = preprocess_midi(midi_produced_path)
        
        sample = {
            'audio': audio_data,
            'midi_original': midi_original_data,
            'midi_produced': midi_produced_data
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def train(workerId, nWorker, filename, runSeed, args):
    parallel = True
    if nWorker == 1:
        parallel = False

    if parallel:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group('nccl', rank=workerId, world_size=nWorker)

    device = torch.device("cuda:" + str(workerId % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    np.random.seed(workerId + int(time.time()))
    torch.manual_seed(workerId + int(time.time()))
    torch.cuda.manual_seed(workerId + int(time.time()))

    if workerId == 0:
        if not os.path.exists(filename):
            print("Initializing the model...")
            conf = None
            if args.modelConf is not None:
                import json
                with open(args.modelConf, 'r') as f:
                    conf = json.load(f)
                    conf = conf[next(iter(conf))]
                    print(conf)

            startEpoch, startIter, model, lossTracker, best_state_dict, optimizer, lrScheduler = initializeCheckpoint(
                TransKun,
                device=device,
                max_lr=args.max_lr,
                weight_decay=args.weight_decay,
                nIter=args.nIter,
                confDict=conf
            )

            save_checkpoint(filename, startEpoch, startIter, model, lossTracker, best_state_dict, optimizer, lrScheduler)

    if parallel:
        dist.barrier()

    startEpoch, startIter, model, lossTracker, best_state_dict, optimizer, lrScheduler = load_checkpoint(TransKun, filename, device)
    print("#{} loaded".format(workerId))

    if workerId == 0:
        print("Loading dataset...")

    dataset = AudioMidiDataset(args.audio_dir, args.midi_original_dir, args.midi_produced_dir)

    if workerId == 0:
        writer = SummaryWriter(filename + ".log")

    globalStep = startIter
    batchSize = args.batchSize
    hopSize = args.hopSize
    chunkSize = args.chunkSize

    gradNormHist = MovingBuffer(initValue=40, maxLen=10000)
    augmentator = None
    if args.augment:
        augmentator = Augmentator(sampleRate=44100)

    for epoch in range(startEpoch, 1000000):
        dataIter = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=args.dataLoaderWorkers)

        if parallel:
            sampler = torch.utils.data.distributed.DistributedSampler(dataIter)
            sampler.set_epoch(epoch)
            dataloader = torch.utils.data.DataLoader(dataIter, batch_size=batchSize, num_workers=args.dataLoaderWorkers, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataIter, batch_size=batchSize, num_workers=args.dataLoaderWorkers, shuffle=True)

        lossAll = []

        for idx, batch in enumerate(dataloader):
            if workerId == 0:
                currentLR = [p["lr"] for p in optimizer.param_groups][0]
                writer.add_scalar('Optimizer/lr', currentLR, globalStep)

            computeStats = False
            if idx % 40 == 0:
                computeStats = True

            t1 = time.time()

            model.train()
            optimizer.zero_grad()

            totalBatch = torch.zeros(1).cuda()
            totalLoss = torch.zeros(1).cuda()
            totalLen = torch.zeros(1).cuda()

            totalGT = torch.zeros(1).cuda()
            totalEst = torch.zeros(1).cuda()
            totalCorrect = torch.zeros(1).cuda()

            totalGTFramewise = torch.zeros(1).cuda()
            totalEstFramewise = torch.zeros(1).cuda()
            totalCorrectFramewise = torch.zeros(1).cuda()

            totalSEVelocity = torch.zeros(1).cuda()
            totalSEOF = torch.zeros(1).cuda()

            audioSlices = batch['audio'].to(device)
            midiOriginal = batch['midi_original'].to(device)
            midiProduced = batch['midi_produced'].to(device)

            audioLength = audioSlices.shape[1] / model.conf.fs

            logp = model.log_prob(audioSlices, midiOriginal, midiProduced)
            loss = (-logp.sum(-1).mean())

            (loss / 50).backward()

            totalBatch += 1
            totalLen += audioLength
            totalLoss += loss.detach()

            if computeStats:
                with torch.no_grad():
                    model.eval()
                    stats = model.computeStats(audioSlices, midiOriginal, midiProduced)
                    stats2 = model.computeStatsMIREVAL(audioSlices, midiOriginal, midiProduced)

            totalGT += stats2["nGT"]
            totalEst += stats2["nEst"]
            totalCorrect += stats2["nCorrect"]
            totalGTFramewise += stats["nGTFramewise"]
            totalEstFramewise += stats["nEstFramewise"]
            totalCorrectFramewise += stats["nCorrectFramewise"]
            totalSEVelocity += stats["seVelocityForced"]
            totalSEOF += stats["seOFForced"]

            if parallel:
                dist.all_reduce(totalLoss.data)
                dist.all_reduce(totalLen.data)
                dist.all_reduce(totalBatch.data)
                if computeStats:
                    dist.all_reduce(totalGT.data)
                    dist.all_reduce(totalEst.data)
                    dist.all_reduce(totalCorrect.data)
                    dist.all_reduce(totalGTFramewise.data)
                    dist.all_reduce(totalEstFramewise.data)
                    dist.all_reduce(totalCorrectFramewise.data)
                    dist.all_reduce(totalSEOF.data)
                    dist.all_reduce(totalSEVelocity.data)

            average_gradients(model, totalLen, parallel)
            loss = totalLoss / totalLen
            curClipValue = gradNormHist.getQuantile(args.gradClippingQuantile)
            totalNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), curClipValue)
            gradNormHist.step(totalNorm.item())

            optimizer.step()
            lrScheduler.step()

            if workerId == 0:
                t2 = time.time()
                print(f"epoch:{epoch} progress:{idx/len(dataloader):.3f} step:0 loss:{loss.item():.4f} gradNorm:{totalNorm.item():.2f} clipValue:{curClipValue:.2f} time:{t2-t1:.2f}")
                writer.add_scalar('Loss/train', loss.item(), globalStep)
                writer.add_scalar('Optimizer/gradNorm', totalNorm.item(), globalStep)
                writer.add_scalar('Optimizer/clipValue', curClipValue, globalStep)
                if computeStats:
                    nGT = totalGT.item() + 1e-4
                    nEst = totalEst.item() + 1e-4
                    nCorrect = totalCorrect.item() + 1e-4
                    precision = nCorrect / nEst
                    recall = nCorrect / nGT
                    f1 = 2 * precision * recall / (precision + recall)
                    print(f"nGT:{nGT} nEst:{nEst} nCorrect:{nCorrect}")

                    writer.add_scalar('Loss/train_f1', f1, globalStep)
                    writer.add_scalar('Loss/train_precision', precision, globalStep)
                    writer.add_scalar('Loss/train_recall', recall, globalStep)

                    nGTFramewise = totalGTFramewise.item() + 1e-4
                    nEstFramewise = totalEstFramewise.item() + 1e-4
                    nCorrectFramewise = totalCorrectFramewise.item() + 1e-4
                    precisionFrame = nCorrectFramewise / nEstFramewise
                    recallFrame = nCorrectFramewise / nGTFramewise
                    f1Frame = 2 * precisionFrame * recallFrame / (precisionFrame + recallFrame)

                    mseVelocity = totalSEVelocity.item() / nGT
                    mseOF = totalSEOF.item() / nGT

                    writer.add_scalar('Loss/train_f1_frame', f1Frame, globalStep)
                    writer.add_scalar('Loss/train_precision_frame', precisionFrame, globalStep)
                    writer.add_scalar('Loss/train_recall_frame', recallFrame, globalStep)
                    writer.add_scalar('Loss/train_mse_velocity', mseVelocity, globalStep)
                    writer.add_scalar('Loss/train_mse_OF', mseOF, globalStep)
                    print(f"f1:{f1} precision:{precision} recall:{recall}")
                    print(f"f1Frame:{f1Frame} precisionFrame:{precisionFrame} recallFrame:{recallFrame}")
                    print(f"mseVelocity:{mseVelocity} mseOF:{mseOF}")

                if math.isnan(loss.item()):
                    exit()
                lossAll.append(loss.item())

                if idx % 400 == 399:
                    save_checkpoint(filename, epoch+1, globalStep+1, model, lossTracker, best_state_dict, optimizer, lrScheduler)
                    print("saved")

            globalStep += 1
            torch.cuda.empty_cache()

        if workerId == 0:
            print("Validating...")
        torch.cuda.empty_cache()

        dataIterVal = DatasetMaestroIterator(dataset, hopSizeInSecond=hopSize, chunkSizeInSecond=chunkSize, notesStrictlyContained=True, seed=runSeed+epoch*100)
        if parallel:
            samplerVal = torch.utils.data.distributed.DistributedSampler(dataIterVal)
            dataloaderVal = torch.utils.data.DataLoader(dataIterVal, batch_size=batchSize, collate_fn=collate_fn, num_workers=0, sampler=samplerVal)
        else:
            dataloaderVal = torch.utils.data.DataLoader(dataIterVal, batch_size=batchSize, collate_fn=collate_fn, num_workers=1, shuffle=True)

        model.eval()
        valResult = doValidation(model, dataloaderVal, parallel=parallel, device=device)

        nll = valResult["meanNLL"]
        f1 = valResult["f1"]

        torch.cuda.empty_cache()

        if workerId == 0:
            lossAveraged = sum(lossAll) / len(lossAll)
            lossAll = []
            lossTracker['train'].append(lossAveraged)
            lossTracker['val'].append(f1)

            print('result:', valResult)

            for key in valResult:
                writer.add_scalar('Loss/val/' + key, valResult[key], epoch)

            if f1 >= max(lossTracker['val']) * 1.00:
                print('best updated')
                best_state_dict = copy.deepcopy(model.state_dict())

            save_checkpoint(filename, epoch+1, globalStep+1, model, lossTracker, best_state_dict, optimizer, lrScheduler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Perform Training")
    parser.add_argument('saved_filename')
    parser.add_argument('--nProcess', help="# of processes for parallel training", required=True, type=int)
    parser.add_argument('--master_addr', help="master address for distributed training", default='127.0.0.1')
    parser.add_argument('--master_port', help='master port number for distributed training', default="29500")
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--midi_original_dir', required=True)
    parser.add_argument('--midi_produced_dir', required=True)
    parser.add_argument('--batchSize', default=2, type=int)
    parser.add_argument('--hopSize', default=10, type=float)
    parser.add_argument('--chunkSize', default=20, type=float)
    parser.add_argument('--dataLoaderWorkers', default=2, type=int)
    parser.add_argument('--gradClippingQuantile', default=0.8, type=float)
    parser.add_argument('--max_lr', default=6e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nIter', default=180000, type=int)
    parser.add_argument('--modelConf', required=False, help="the path to the model conf file")
    parser.add_argument('--augment', action="store_true", help="do data augmentation")

    args = parser.parse_args()

    num_processes = args.nProcess
    saved_filename = args.saved_filename

    if num_processes == 1:
        train(0, 1, saved_filename, int(time.time()), args)
    else:
        mp.spawn(fn=train, args=(num_processes, saved_filename, int(time.time()), args), nprocs=num_processes, join=True, daemon=False)
