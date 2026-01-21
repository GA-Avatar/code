# import argparse
# from config import cfg
# import torch
# from base import Trainer

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--subject_id', type=str, dest='subject_id')
#     parser.add_argument('--continue', dest='continue_train', action='store_true')

#     args = parser.parse_args()
#     assert args.subject_id, "Please set subject ID"
#     return args

# def main():
#     args = parse_args()
#     cfg.set_args(args.subject_id, args.continue_train)

#     trainer = Trainer()
#     trainer._make_batch_generator()
#     trainer._make_model()
    
#     for epoch in range(trainer.start_epoch, cfg.end_epoch):
#         trainer.tot_timer.tic()
#         trainer.read_timer.tic()

#         for itr, data in enumerate(trainer.batch_generator):
#             trainer.read_timer.toc()
#             trainer.gpu_timer.tic()
            
#             # set stage
#             cur_itr = epoch * len(trainer.batch_generator) + itr
#             cfg.set_stage(cur_itr)

#             # set learning rate
#             tot_itr = cfg.end_epoch * len(trainer.batch_generator)
#             trainer.set_lr(cur_itr, tot_itr)
           
#             # forward
#             trainer.optimizer.zero_grad()
#             loss = trainer.model(data, 'train')
#             loss = {k:loss[k].mean() for k in loss}

#             # backward
#             sum(loss[k] for k in loss).backward()
            
#             # update
#             trainer.optimizer.step()
            
#             # log
#             trainer.gpu_timer.toc()
#             screen = [
#                 'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
#                 'speed: %.2f(%.2fs r%.2f)s/itr' % (
#                     trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
#                 '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
#                 ]
#             screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
#             trainer.logger.info(' '.join(screen))
#             print(cfg.model_dir)

#             trainer.tot_timer.toc()
#             trainer.tot_timer.tic()
#             trainer.read_timer.tic()
#             cur_itr += 1

#         # save model
#         trainer.save_model({
#             'epoch': epoch,
#             'network': trainer.model.module.state_dict(),
#             'optimizer': trainer.optimizer.state_dict(),
#         }, epoch)
   
# if __name__ == "__main__":
#     main()
import argparse
from config import cfg
import torch
from base import Trainer
from torch.utils.tensorboard import SummaryWriter   # ✅ 加tensorboard

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs',help='Directory to save results and logs')
    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id, args.continue_train, args.output_dir)

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # ✅ 初始化 TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.model_dir + '/logs')

    global_step = 0  # 记录全局迭代次数

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, data in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            # set stage
            cur_itr = epoch * len(trainer.batch_generator) + itr
            cfg.set_stage(cur_itr)

            # set learning rate
            tot_itr = cfg.end_epoch * len(trainer.batch_generator)
            trainer.set_lr(cur_itr, tot_itr)
           
            # forward
            trainer.optimizer.zero_grad()
            # loss = trainer.model(data, 'train')
            # loss = trainer.model(data, 'train', cur_itr)
            loss = trainer.model(data, 'train', epoch)
            loss = {k: loss[k].mean() for k in loss}

            # backward
            total_loss = sum(loss[k] for k in loss)
            total_loss.backward()
            
            # update
            trainer.optimizer.step()
            
            # ✅ 把 loss 写入 TensorBoard
            for k, v in loss.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            writer.add_scalar('Loss/total', total_loss.item(), global_step)

            global_step += 1

            # log
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))
            print(cfg.model_dir)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        # # save model
        # trainer.save_model({
        #     'epoch': epoch,
        #     'network': trainer.model.module.state_dict(),
        #     'optimizer': trainer.optimizer.state_dict(),
        # }, epoch)
        # save model
        if epoch in [24]:  # 在epoch = 15, 20, 24时保存
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.module.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch)

    # ✅ 训练结束关闭 writer
    writer.close()
   
if __name__ == "__main__":
    main()
