import subprocess

def main():

    cmds = [
        "CUDA_VISIBLE_DEVICES=1 python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train --use_mc_dropout --use_reward_norm --dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 --roll_num 16 --mc_samples 16 --buffer_capacity 10000 --replay_prob 0.2 --properties druglikeness --surrogate_threshold 0.2 --surrogate_uncertainty_threshold 0.5",
        "CUDA_VISIBLE_DEVICES=1 python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train --use_mc_dropout --use_reward_norm --dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 --roll_num 16 --mc_samples 16 --buffer_capacity 10000 --replay_prob 0.2 --properties solubility --surrogate_threshold 0.2 --surrogate_uncertainty_threshold 0.5",
        "CUDA_VISIBLE_DEVICES=1 python main.py --gen_pretrain --dis_pretrain --dis_wgan --dis_minibatch --adversarial_train --use_mc_dropout --use_reward_norm --dataset_name QM9 --max_len 60 --generated_num 5000 --gen_train_size 4800 --roll_num 16 --mc_samples 16 --buffer_capacity 10000 --replay_prob 0.2 --properties synthesizability --surrogate_threshold 0.2 --surrogate_uncertainty_threshold 0.5"
        ]

    for cmd in cmds:
        print(f"🚀 Running: {cmd}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()