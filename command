srun --nodelist=slurm0-a3-ghpc-0 --time=48:00:00 --gpus-per-node=1 --cpus-per-task=16 --mem=100GB --pty bash -i

source ~/miniconda3/etc/profile.d/conda.sh
conda activate textprocess

sbatch --nodelist=slurm0-a3-ghpc-0 --cpus-per-task=32 tokenize_dir_sb.sh

sbatch --nodelist=slurm0-a3-ghpc-0 --cpus-per-task=16 tokenize_dir_sb.sh
sbatch --nodelist=slurm0-a3-ghpc-1 --cpus-per-task=8 make_pack_jsonl_sb.sh
sbatch --nodelist=slurm0-a3-ghpc-0 --cpus-per-task=2 combinning_sb_sh

/storage5/shared/corpus/synthetic/SyntheticTexts/flags

squeue
scontrol show job 236
squeue -u ext_k_nishizawa_p_gmail_com
tail -f slurm-

for file in /storage5/shared/p2_corpus/before_tokenize_jsonl/ xxx ; do mv "$file" "${file}.jsonl"; done

cat ./*.jsonl > combined.jsonl
split -l 500000 combined.jsonl output_prefix

for file in ./* ; do mv "$file" "${file}.jsonl"; done

watch -n 2 "squeue | tail -n 20"
