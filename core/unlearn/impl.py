# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import copy
# from torch.optim.swa_utils import AveragedModel
# import time
#
# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T
#
#     def forward(self, y_s, y_t):
#         p_s = F.log_softmax(y_s / self.T, dim=1)
#         p_t = F.softmax(y_t / self.T, dim=1)
#         loss = F.kl_div(p_s, p_t, reduction="sum") * (self.T ** 2) / y_s.shape[0]
#         return loss
#
#
# def adjust_learning_rate(
#         epoch, lr_decay_epochs, lr_decay_rate, sgda_learning_rate, optimizer
# ):
#     """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
#     steps = np.sum(epoch > np.asarray(lr_decay_epochs))
#     new_lr = sgda_learning_rate
#     if steps > 0:
#         new_lr = sgda_learning_rate * (lr_decay_rate ** steps)
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = new_lr
#     return new_lr
#
#
# def param_dist(model, swa_model, p):
#     # This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
#     dist = 0.0
#     for p1, p2 in zip(model.parameters(), swa_model.parameters()):
#         dist += torch.norm(p1 - p2, p="fro")
#     return p * dist
#
#
# def _iterative_unlearn_impl(unlearn_iter_func):
#     def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
#         decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
#         # if args.rewind_epoch != 0:
#         #     initialization = torch.load(
#         #         args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
#         #     )
#         #     current_mask = extract_mask(model.state_dict())
#         #     remove_prune(model)
#         #     model.load_state_dict(initialization, strict=True)
#         #     prune_model_custom(model, current_mask)
#
#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             args.theta_lr,
#             momentum=args.momentum,
#             weight_decay=args.weight_decay,
#         )
#
#         if args.imagenet_arch and args.unlearn == "retrain":
#             lambda0 = (
#                 lambda cur_iter: (cur_iter + 1) / args.warmup
#                 if cur_iter < args.warmup
#                 else (
#                         0.5
#                         * (
#                                 1.0
#                                 + np.cos(
#                             np.pi
#                             * (
#                                     (cur_iter - args.warmup)
#                                     / (args.unlearn_epochs - args.warmup)
#                             )
#                         )
#                         )
#                 )
#             )
#             scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
#         else:
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(
#                 optimizer, milestones=decreasing_lr, gamma=0.1
#             )  # 0.1 is fixed
#
#         if args.arch == "swin_t":
#             optimizer = torch.optim.Adam(model.parameters(), args.theta_lr)
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, args.unlearn_steps
#             )
#
#         # if args.rewind_epoch != 0:
#         #     # learning rate rewinding
#         #     for _ in range(args.rewind_epoch):
#         #         scheduler.step()
#
#         if args.unlearn == "scrub":
#             teacher = copy.deepcopy(model)
#             student = copy.deepcopy(model)
#
#             model_t = copy.deepcopy(teacher)
#             model_s = copy.deepcopy(student)
#             module_list = nn.ModuleList([])
#             module_list.append(model_s)
#             trainable_list = nn.ModuleList([])
#             trainable_list.append(model_s)
#             criterion_cls = criterion
#             criterion_div = DistillKL(args.T)
#             criterion_kd = DistillKL(args.T)
#             module_list.append(model_t)
#             beta = args.scrub_beta
#
#             def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
#                 return (1 - beta) * averaged_model_parameter + beta * model_parameter
#
#             swa_model = torch.optim.swa_utils.AveragedModel(
#                 model_s, avg_fn=avg_fn
#             )
#             swa_model.cuda()
#
#             optimizer = torch.optim.SGD(
#                 trainable_list.parameters(),
#                 args.theta_lr,
#                 momentum=args.momentum,
#                 weight_decay=args.weight_decay,
#             )
#
#             for module in module_list:
#                 module.train()
#             module_list[-1].eval()
#
#             eval_model_s = module_list[0]
#             eval_model_t = module_list[-1]
#
#             for epoch in range(0, args.unlearn_steps):
#                 start_time = time.time()
#
#                 print(
#                     "Epoch #{}, Learning rate: {}".format(
#                         epoch, optimizer.state_dict()["param_groups"][0]["lr"]
#                     )
#                 )
#
#                 unlearn_iter_func(
#                     data_loaders, model_s, model_t, eval_model_s, eval_model_t, swa_model,
#                     criterion_cls, criterion_div, criterion_kd,
#                     optimizer, epoch, args, mask, **kwargs
#                 )
#                 print("one epoch duration:{}".format(time.time() - start_time))
#             model.load_state_dict(model_s.state_dict())
#
#         else:
#             for epoch in range(0, args.unlearn_steps):
#                 start_time = time.time()
#
#                 print(
#                     "Epoch #{}, Learning rate: {}".format(
#                         epoch, optimizer.state_dict()["param_groups"][0]["lr"]
#                     )
#                 )
#
#                 train_acc = unlearn_iter_func(
#                     data_loaders, model, criterion, optimizer, epoch, args, mask, **kwargs
#                 )
#                 scheduler.step()
#
#                 print("one epoch duration:{}".format(time.time() - start_time))
#
#     return _wrapped
#
#
# def iterative_unlearn(func):
#
#     return _iterative_unlearn_impl(func)



