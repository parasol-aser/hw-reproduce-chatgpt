# Reproduce GPT-2 (124M)

**Reference:** https://github.com/karpathy/llm.c/discussions/481

### Setup on Grace
```
module load GCC/12.3
module load CUDA/12.3
```
```
cd $SCRATCH
git clone https://github.com/karpathy/llm.c.git
cd llm.c
make train_gpt2cu
```

### Training and Evaluation Data 

- Training: FineWeb dataset 10B tokens (takes ~1H to download)

```
python dev/data/fineweb.py --version 10B
```

- Evaluation: HellaSwag (70K commonsense questions)
```
python dev/data/hellaswag.py
```

- *Alternatively, add a soft link to our shared data folder (pre-downloaded data for both training and evaluation)*

```
rm -rf dev/data
ln -s /scratch/group/csce689609/data dev/
```

### Training Test for 10mins
```
srun --nodes=1 --cpus-per-task=32 --mem=128g --gres=gpu:a100:1  --time=00:10:00 --pty bash -i
```
```
./train_gpt2cu     -i "dev/data/fineweb10B/fineweb_train_*.bin"     -j "dev/data/fineweb10B/fineweb_val_*.bin"     -o log124M     -e "d12"     -b 32 -t 1024     -d 524288     -r 0     -z 1     -c 0.1     -l 0.0006     -q 0.0     -u 700     -n 5000     -v 250 -s 20000     -h 1
```

### Training for 24h
```
sbatch train_gpt2.slurm
```

train_gpt2.slurm:
```
#!/bin/bash
#SBATCH --job-name=gpt2_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=32
#SBATCH --time=26:00:00              #Request 26 hours (2 extra hours)
#SBATCH --mem=128GB                  #Request 128GB per node
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --gres=gpu:a100:1            #Request one A100 GPU to use

#SBATCH --output=gpt2_train.%j.log            #Redirect stdout/err to file

# Run the training script
./train_gpt2cu     -i "dev/data/fineweb10B/fineweb_train_*.bin"     -j "dev/data/fineweb10B/fineweb_val_*.bin"     -o log124M     -e "d12"     -b 32 -t 1024     -d 524288     -r 0     -z 1     -c 0.1     -l 0.0006     -q 0.0     -u 700     -n 5000     -v 250 -s 20000     -h 1
```

Logs:
```
step    1/19560 | loss 11.010252 (+nanz)| norm 15.1374 (+nanz)| lr 8.57e-07 | 4688.25 ms | 28.8% bf16 MFU | 111830 tok/s
...
step 19559/19560 | loss 3.314194 (+0.19z)| norm 0.2776 (+3.74z)| lr 1.79e-11 | 4339.91 ms | 31.1% bf16 MFU | 124285 tok/s
step 19560/19560 | loss 3.252871 (-1.21z)| norm 0.2281 (+0.02z)| lr 0.00e+00 | 4367.96 ms | 30.9% bf16 MFU | 124072 tok/s
val loss 3.265413
evaluating HellaSwag: 770/837
HellaSwag: 3151/10042 = 0.313782

generating:
---
The name of Sam Ray Dalsman is Recappable because he is the same male as Hogwood, John Hand, Arden Moses, George Haddock, and Stephen Ennies. An unbounded band that was banned by American governments to such offensive and subversive acts as Homopoglands is tainted with
---
Writing checkpoint at step 19560
Writing model to log124M/model_00019560.bin
Writing state to log124M/state_00019560_00000.bin
total average iteration time: 4179.953030 ms
```

Evaluation results:
```
----------------------------------------
arc_challenge_25shot.json      : 22.0137
gsm8k_5shot.json               : 0.0758
hellaswag_10shot.json          : 31.3782
mmlu_5shot.json                : 25.6644
truthfulqa_0shot.json          : 42.6879
winogrande_5shot.json          : 49.8816
----------------------------------------
Average Score                  : 28.6169
```

**Baseline HellaSwag accuracy:** 31.3782
