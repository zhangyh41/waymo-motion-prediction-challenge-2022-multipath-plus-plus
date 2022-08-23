import torch
import numpy as np
from model.eval_metrics import PredictionEvaluator


def compute_metrics(batch, intention, score, pred_traj, target_time):
    gt_pos = batch.gt_pos
    if target_time != gt_pos.shape[1]/10:
        target_ts = target_time * 10
        gt_pos = gt_pos[:, :target_ts]
        pred_traj = pred_traj[:, :, :target_ts]
        intention = pred_traj[:, :, target_ts-1]

    # calculate velocity for waymo_mr
    if hasattr(batch, 'vector'):
        start_pos = batch.vector[:, 0, -1, :2] #[B, 2]
        end_pos = batch.vector[:, 0, -1, 2:4] #[B, 2]
    else:
        start_pos = batch.actor_vector[:, 0, -1, :2] #[B, 2]
        end_pos = batch.actor_vector[:, 0, -1, 2:4] #[B, 2]
    waymo_thres_velocity = 10 * torch.linalg.norm(end_pos - start_pos, dim=-1) #[B]
    waymo_thres_t = gt_pos.shape[1]/10

    # top1    
    top1_min_fde, zero_idxes = PredictionEvaluator.get_min_fde(intention[:, :1], gt_pos[:, -1])
    top1_min_ade, _ = PredictionEvaluator.get_min_ade(pred_traj[:, :1], gt_pos)
    top1_mr = PredictionEvaluator.get_mr(top1_min_fde)
    top1_waymo_mr = PredictionEvaluator.get_waymo_mr(intention[:, :1], gt_pos, waymo_thres_t, waymo_thres_velocity)
    
    # topk
    topk_min_fde, min_fde_idx = PredictionEvaluator.get_min_fde(intention, gt_pos[:, -1])
    topk_min_ade, min_ade_idx = PredictionEvaluator.get_min_ade(pred_traj, gt_pos)
    topk_argo_min_ade = PredictionEvaluator.get_argo_min_ade(pred_traj, gt_pos, min_fde_idx)
    topk_mr = PredictionEvaluator.get_mr(topk_min_fde)
    topk_waymo_mr = PredictionEvaluator.get_waymo_mr(intention, gt_pos, waymo_thres_t, waymo_thres_velocity)
    brier_min_fde = PredictionEvaluator.get_brier_min_fde(topk_min_fde, min_fde_idx, score)
    brier_min_ade = PredictionEvaluator.get_brier_min_ade(topk_min_ade, min_ade_idx, score)

    result = {'brier_min_fde': brier_min_fde.cpu().numpy().tolist(),
              'brier_min_ade': brier_min_ade.cpu().numpy().tolist(),
              'topk_min_fde': topk_min_fde.cpu().numpy().tolist(),
              'min_fde_idx': min_fde_idx.cpu().numpy().tolist(),
              'topk_min_ade': topk_min_ade.cpu().numpy().tolist(),
              'min_ade_idx': min_ade_idx.cpu().numpy().tolist(),
              'topk_argo_min_ade': topk_argo_min_ade.cpu().numpy().tolist(),
              'topk_mr': [topk_mr],
              'topk_waymo_mr': [topk_waymo_mr],
              'top1_min_fde': top1_min_fde.cpu().numpy().tolist(),
              'top1_min_ade': top1_min_ade.cpu().numpy().tolist(),
              'top1_mr': [top1_mr],
              'top1_waymo_mr': [top1_waymo_mr]}
    
    return result

def collate_compute_metrics(res_list):
    results = {}
    results_list = {}
    for key in res_list[0]:
        results[key] = {}
        results_list[key] = {}
        for res in res_list:
            for metric_key in res[key]:
                if metric_key not in results_list[key]:
                    results_list[key][metric_key] = []
                results_list[key][metric_key].extend(res[key][metric_key])
        for metric_key in res_list[0][key]:
            if metric_key != 'min_fde_idx' and metric_key != 'min_ade_idx':
                results[key][metric_key] = np.mean(results_list[key][metric_key])
    
    return results, results_list