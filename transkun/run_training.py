import argparse
import os
import time
import torch.multiprocessing as mp
from train_revised import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the training process for TransKun model.")
    
    parser.add_argument('--saved_filename', required=True, help='Path to save the checkpoint file.')
    parser.add_argument('--nProcess', type=int, default=1, help='# of processes for parallel training.')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master address for distributed training.')
    parser.add_argument('--master_port', type=str, default='29500', help='Master port number for distributed training.')
    parser.add_argument('--audio_dir', required=True, help='Directory containing audio files.')
    parser.add_argument('--midi_original_dir', required=True, help='Directory containing original MIDI files.')
    parser.add_argument('--midi_produced_dir', required=True, help='Directory containing produced MIDI files.')
    parser.add_argument('--batchSize', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--hopSize', type=float, default=10, help='Hop size in seconds.')
    parser.add_argument('--chunkSize', type=float, default=20, help='Chunk size in seconds.')
    parser.add_argument('--dataLoaderWorkers', type=int, default=2, help='# of workers for data loading.')
    parser.add_argument('--gradClippingQuantile', type=float, default=0.8, help='Gradient clipping quantile.')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer.')
    parser.add_argument('--nIter', type=int, default=180000, help='# of iterations.')
    parser.add_argument('--modelConf', type=str, help='Path to the model configuration file.')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation.')

    args = parser.parse_args()

    if args.nProcess == 1:
        train(0, 1, args.saved_filename, int(time.time()), args)
    else:
        mp.spawn(train, args=(args.nProcess, args.saved_filename, int(time.time()), args), nprocs=args.nProcess, join=True)
