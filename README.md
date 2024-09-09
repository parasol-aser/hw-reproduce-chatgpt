# Reproduce GPT-2 (124M)

### Setup on Grace
```
module load GCC/12.3
module load CUDA/12.3.0
```
```
git clone https://github.com/karpathy/llm.c.git
cd llm.c
make train_gpt2cu
```

### Training Data and Evaluation 

**Reference:** https://github.com/karpathy/llm.c/discussions/481

- FineWeb dataset 10B tokens (takes ~1H to download)

```
python dev/data/fineweb.py --version 10B
```

- HellaSwag eval (70K commonsense questions)
```
python dev/data/hellaswag.py
```

### Training Test for 10mins
```
srun --nodes=1 --cpus-per-task=8 --mem=360g --gres=gpu:a100:1  --time=00:10:00 --pty bash -i
./train_gpt2cu     -i "dev/data/fineweb10B/fineweb_train_*.bin"     -j "dev/data/fineweb10B/fineweb_val_*.bin"     -o log124M     -e "d12"     -b 32 -t 1024     -d 524288     -r 0     -z 1     -c 0.1     -l 0.0006     -q 0.0     -u 700     -n 5000     -v 250 -s 20000     -h 1
```

### Training for 24H Max
```
sbatch train_gpt2.slurm
```
Logs:
```
step    1/19560 | loss 11.010252 (+nanz)| norm 15.1374 (+nanz)| lr 8.57e-07 | 4688.25 ms | 28.8% bf16 MFU | 111830 tok/s
...
step   80/19560 | loss 7.512386 (+nanz)| norm 1.1204 (+nanz)| lr 6.86e-05 | 4187.58 ms | 32.2% bf16 MFU | 125356 tok/s
...
```

**Baseline HellaSwag accuracy:** 29.9
