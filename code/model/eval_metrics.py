import torch
from data.intention_transform8 import IntentionTransform8

class PredictionEvaluator():
    @staticmethod
    def get_min_fde(pred_target, gt_target):
        r'''
        pred_target in shape [B, K, 2]
        gt_target in shape [B,2]

        return:
        min_fde in shape [B]
        min_idx in shape [B]
        '''

        dist = torch.linalg.norm(pred_target - gt_target.unsqueeze(1),
                                 dim=-1) #[B, K]
        min_fde, min_idx = torch.min(dist, dim=-1) #[B]

        return min_fde, min_idx

    @staticmethod
    def get_brier_min_fde(min_fde_topk, min_fde_idx, score):
        r'''
        min_fde_topk in shape [B]
        min_fde_idx in shape [B]
        score in shape [B, K]
        '''
        pruned_probabilities = score / torch.sum(score, dim=1).unsqueeze(1) #[B, K]
        p = pruned_probabilities[torch.arange(len(min_fde_idx)), min_fde_idx] #[B]
        brier_min_fde = min_fde_topk + (1-p) ** 2
        return brier_min_fde


    @staticmethod
    def get_mr(min_fde, threshold=2.0):
        r'''
        min_fde in shape [B]
        '''

        mr = torch.sum(min_fde > threshold).item() / len(min_fde)

        return mr

    @staticmethod
    def get_waymo_mr(pred_target, gt_pos, t, velocity):
        r'''
        pred_target in shape [B, K, 2]
        gt_pos in shape [B, T, 2]
        min_fde_idx in shape [B]
        velocity in shape [B]
        '''
        
        miss_num = 0
        B = pred_target.shape[0]
        for i in range(B):
            gt_rotm = IntentionTransform8.rotmFromVect(gt_pos[i, -2].cpu().numpy(), gt_pos[i, -1].cpu().numpy())
            gt_rotm = torch.from_numpy(gt_rotm).to(gt_pos.device)
            fde_xy = torch.matmul(pred_target[i] - gt_pos[i, -1], gt_rotm)
            fde_xy = torch.abs(fde_xy)

            threshold_x = threshold_y = 2.0
            if t == 3:
                threshold_x = 2.0
                threshold_y = 1.0
            elif t == 5:
                threshold_x = 3.6
                threshold_y = 1.8
            elif t == 8:
                threshold_x = 6
                threshold_y = 3

            v = velocity[i]
            scale = 1
            if v < 1.4:
                scale = 0.5
            elif v < 11:
                scale = 0.5 + 0.5 * (v - 1.4) / (11 - 1.4)
            
            hit =  (fde_xy[:, 0] <= (scale * threshold_x)) & (fde_xy[:, 1] <= (scale * threshold_y))
            if not hit.any():
                miss_num += 1
        mr = miss_num / B
        return mr


    @staticmethod
    def get_min_ade(pred_traj, gt_traj):
        r'''
        pred_traj in shape [B, K, T, 2]
        gt_traj in shape [B, T, 2]

        return:
        min_ade in shape [B]
        '''

        p_dist = torch.linalg.norm(pred_traj - gt_traj.unsqueeze(1), dim=-1) #[B, K, 30]
        dist = torch.mean(p_dist, dim=-1) #[B, K]
        min_ade, min_idx = torch.min(dist, dim=-1) #[B]

        return min_ade, min_idx

    @staticmethod
    def get_brier_min_ade(min_ade_topk, min_ade_idx, score):
        r'''
        min_ade_topk in shape [B]
        min_ade_idx in shape [B]
        score in shape [B, K]
        '''
        pruned_probabilities = score / torch.sum(score, dim=1).unsqueeze(1) #[B, K]
        p = pruned_probabilities[torch.arange(len(min_ade_idx)), min_ade_idx] #[B]
        brier_min_ade = min_ade_topk + (1-p) ** 2
        return brier_min_ade


    @staticmethod
    def get_argo_min_ade(pred_traj, gt_traj, min_fde_idx):
        r'''
        pred_traj in shape [B, K, T, 2]
        gt_traj in shape [B, T, 2]
        min_fde_idx in shape [B]

        return:
        argo_min_ade in shape [B]
        '''

        p_dist = torch.linalg.norm(pred_traj[torch.arange(len(min_fde_idx)), min_fde_idx] - gt_traj, dim=-1) #[B, 30]
        argo_min_ade = torch.mean(p_dist, dim=-1) #[B]

        return argo_min_ade