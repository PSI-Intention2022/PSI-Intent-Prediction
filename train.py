import collections

from test import validate_intent
import torch
import numpy as np
import os

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train_intent(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer):
    pos_weight = torch.tensor(args.intent_positive_weight).to(device) # n_neg_class_samples(5118)/n_pos_class_samples(11285)
    criterions = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight).to(device),
        'MSELoss': torch.nn.MSELoss(reduction='none').to(device),
        'BCELoss': torch.nn.BCELoss().to(device),
        'CELoss': torch.nn.CrossEntropyLoss(),
    }
    epoch_loss = {'loss_intent': [], 'loss_traj': []}

    for epoch in range(args.epochs):
        niters = len(train_loader)
        recorder.train_epoch_reset(epoch, niters)
        epoch_loss = train_intent_epoch(epoch, model, optimizer, criterions, epoch_loss, train_loader, args, recorder, writer)
        scheduler.step()

        if epoch % 1 == 0:
            print(f"Train epoch {epoch}/{args.epochs} | epoch loss: "
                  f"loss_intent = {np.mean(epoch_loss['loss_intent']): .4f}")

        if (epoch + 1) % args.val_freq == 0:
            print(f"Validate at epoch {epoch}")
            niters = len(val_loader)
            recorder.eval_epoch_reset(epoch, niters)
            validate_intent(epoch, model, val_loader, args, recorder, writer)

            # result_path = os.path.join(args.checkpoint_path, 'results', f'epoch_{epoch}')
            # if not os.path.isdir(result_path):
            #     os.makedirs(result_path)
            # recorder.save_results(prefix='')
            # torch.save(model.state_dict(), result_path + f'/state_dict.pth')

        torch.save(model.state_dict(), args.checkpoint_path + f'/latest.pth')


def train_intent_epoch(epoch, model, optimizer, criterions, epoch_loss, dataloader, args, recorder, writer):
    model.train()
    batch_losses = collections.defaultdict(list)

    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        optimizer.zero_grad()
        intent_logit = model(data)
        # intent_pred: sigmoid output, (0, 1), bs
        # traj_pred: logit, bs x ts x 4

        # 1. intent loss
        if args.intent_type == 'mean' and args.intent_num == 2: # BCEWithLogitsLoss
            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)

            gt_disagreement = data['disagree_score'][:, args.observe_length]
            gt_consensus = (1 - gt_disagreement).to(device)

            loss_intent = 0
            if 'bce' in args.intent_loss:
                loss_intent_bce = criterions['BCEWithLogitsLoss'](intent_logit, gt_intent)

                if args.intent_disagreement != -1.0:
                    if args.ignore_uncertain:
                        mask = (gt_consensus > args.intent_disagreement) * gt_consensus
                    else:
                        mask = gt_consensus
                    loss_intent_bce = torch.mean(torch.mul(mask, loss_intent_bce))
                else: # -1.0, not use reweigh and filter
                    loss_intent_bce = torch.mean(loss_intent_bce)
                batch_losses['loss_intent_bce'].append(loss_intent_bce.item())
                loss_intent += loss_intent_bce

            if 'mse' in args.intent_loss:
                loss_intent_mse = criterions['MSELoss'](gt_intent_prob, torch.sigmoid(intent_logit))

                if args.intent_disagreement != -1.0:
                    mask = (gt_consensus > args.intent_disagreement) * gt_consensus
                    loss_intent_mse = torch.mean(torch.mul(mask, loss_intent_mse))
                else: # -1.0, not use reweigh and filter
                    loss_intent_mse = torch.mean(loss_intent_mse)

                batch_losses['loss_intent_mse'].append(loss_intent_mse.item())

                loss_intent += loss_intent_mse

        loss = args.loss_weights['loss_intent'] * loss_intent

        loss.backward()
        optimizer.step()

        # Record results
        batch_losses['loss'].append(loss.item())
        batch_losses['loss_intent'].append(loss_intent.item())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters} - "
                  f"loss_intent = {np.mean(batch_losses['loss_intent']): .4f}")
        intent_prob = torch.sigmoid(intent_logit)
        recorder.train_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                           gt_intent_prob.detach().cpu().numpy(),
                                           intent_prob.detach().cpu().numpy(),
                                           loss.item(), loss_intent.item())

    epoch_loss['loss_intent'].append(np.mean(batch_losses['loss_intent']))

    recorder.train_intent_epoch_calculate(writer)
    # write scalar to tensorboard
    writer.add_scalar(f'LearningRate', optimizer.param_groups[-1]['lr'], epoch)
    for key, val in batch_losses.items():
        writer.add_scalar(f'Losses/{key}', np.mean(val), epoch)

    return epoch_loss