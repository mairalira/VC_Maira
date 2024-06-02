# -*- encoding: utf-8 -*-
'''
@reference: Seul-Ki Yeom et al., "Pruning by explaining: a novel criterion for deep neural network pruning," Pattern Recognition, 2020.
@author: Seul-Ki Yeom, Philipp Seegerer, Sebastian Lapuschkin, Alexander Binder, Simon Wiedemann, Klaus-Robert Müller, Wojciech Samek
'''

from __future__ import print_function
import argparse
import numpy as np
import torch
import os

from modules.network import ResNet18, ResNet50, VGG_Alex
import modules.prune_resnet as modules_resnet
import modules.prune_vgg as modules_vgg

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch VGG16 based ImageNet')

    parser.add_argument('--arch', default='vgg16', metavar='ARCH',
                        help='model architecture: resnet18, resnet50, vgg16, alexnet')
    parser.add_argument('--train-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 20)')
    parser.add_argument('--trialnum', type=int, default=1, metavar='N',
                        help='trial number (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # parser.add_argument('--train', action='store_true', help='training data')
    parser.add_argument('--prune', action='store_true', help='pruning model')
    # parser.add_argument('--relevance', action='store_true', help='Compute relevances')
    parser.add_argument('--norm', action='store_true', help='add normalization')
    parser.add_argument('--resume', type=bool, default=True, metavar='N',
                        help='if we have pretrained model')
    parser.add_argument('--train', type=bool, default=False, metavar='N',
                        help='training data')
    # parser.add_argument('--prune', type=bool, default=True, metavar='N',
    #                   help='pruning model')
    parser.add_argument('--method-type', type=str, default='lrp', metavar='N',
                        help='model architecture selection: grad/taylor/weight/lrp')

    parser.add_argument('--total-pr', type=float, default=9.001 / 10.0, metavar='M',
                        help='Total pruning rate')
    parser.add_argument('--pr-step', type=float, default=0.05, metavar='M',
                        help='Pruning step: 0.05 (5% for each step)')

    parser.add_argument('--data-type', type=str, default='cifar10', metavar='N',
                        help='model architecture selection: cifar10/imagenet/catsvsdogs')
    parser.add_argument('--save-dir', type=str, default='model', metavar='N',
                        help='saved directory')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

if __name__ == '__main__':
    args = get_args()

    if args.arch.lower() not in ['alexnet', 'vgg16', 'resnet18', 'resnet50']:
        raise ValueError('Unsupported architecture specified: {}'.format(args.arch))
        
    model = {
        'alexnet': VGG_Alex(arch='alexnet'),
        'vgg16': VGG_Alex(arch='vgg16'),
        'resnet18': ResNet18(),
        'resnet50': ResNet50(),
    }[args.arch.lower()]

    if args.resume:
        save_loc = f"./checkpoint/{args.arch}_{args.data_type}_ckpt.pth"
        print(f'Saving model checkpoint to: {save_loc}') #to confirm checkpoint address
        try: 
            torch.save(model.state_dict(), save_loc)
            print(f'Model saved successfully at {save_loc}')
        except Exception as e:
            print(f'Error saving model: {e}')
        #model.load_state_dict(torch.load(save_loc)) if args.cuda else model.load_state_dict(torch.load(save_loc))

    if args.cuda:
        model = model.cuda()

    if args.arch.lower() in ['resnet18', 'resnet50']:
        fine_tuner = modules_resnet.PruningFineTuner(args, model)
    elif args.arch.lower() in ['vgg16', 'alexnet']:
        fine_tuner = modules_vgg.PruningFineTuner(args, model)

    if args.train:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print(f'Start training! Dataset: {args.data_type}, Architecture: {args.arch}, Epoch: {args.epochs}')
        fine_tuner.train(optimizer=optimizer,epochs=args.epochs)

    if args.prune:
        print(f'Start pruning! Dataset: {args.data_type}, Architecture: {args.arch}, Pruning Method: {args.method_type},'
              f' Total Pruning rate: {args.total_pr}, Pruning step: {args.pr_step}')
        fine_tuner.prune()
    else:
        # If pruning is not performed, ensure initial evaluation and results saving
        print(f'Evaluating model without pruning.  Dataset: {args.data_type}, Architecture: {args.arch}')
        test_accuracy, test_loss = fine_tuner.test()
        flop_value, param_value = fine_tuner.calculate_flops_params()

        # Initialize the DataFrame to store results
        fine_tuner.df = pd.DataFrame(columns=["ratio_pruned", "test_acc", "test_loss", "flops", "params"])
        fine_tuner.dt = pd.DataFrame(columns=["epoch", "train_acc", "train_loss"])

        fine_tuner.df.loc[fine_tuner.COUNT_ROW] = pd.Series({"ratio_pruned": 0.0,
                                                             "test_acc": test_accuracy,
                                                             "test_loss": test_loss,
                                                             "flops": flop_value,
                                                             "params": param_value})
        fine_tuner.COUNT_ROW += 1

        # Define the results file paths
        results_file = f"{args.save_dir}/scenario1_results_{args.data_type}_{args.arch}_{args.method_type}_trial{args.trialnum:02d}.csv"
        results_file_train = f"{args.save_dir}/scenario1_train_{args.data_type}_{args.arch}_{args.method_type}_trial{args.trialnum:02d}.csv"

        # Save the initial evaluation results
        fine_tuner.df.to_csv(results_file)
        fine_tuner.dt.to_csv(results_file_train)

# Test model on CIFAR-10
#print("Testing model on CIFAR-10...")
#fine_tuner.setup_dataloaders(dataset_type='cifar10')  # Setup CIFAR-10 data loaders
#cifar10_accuracy, cifar10_loss = fine_tuner.test()
#print(f"CIFAR-10 Accuracy: {cifar10_accuracy}, CIFAR-10 Loss: {cifar10_loss}")

# Test model on Cats vs. Dogs
#print("Testing model on Cats vs. Dogs...")
#fine_tuner.setup_dataloaders(dataset_type="catsvsdogs")  # Setup Cats vs. Dogs data loaders
#catsvsdogs_accuracy, catsvsdogs_loss = fine_tuner.test()
#print(f"Cats vs. Dogs Accuracy: {catsvsdogs_accuracy}, Cats vs. Dogs Loss: {catsvsdogs_loss}")
