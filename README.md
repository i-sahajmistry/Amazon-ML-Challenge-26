# IIT Delhi HPC (PADUM) Setup — Amazon ML Challenge 2026

Tested and working end-to-end. Replace `{kerberos_id}` with your own Kerberos username everywhere.

Faculty supervisor: `raunakbh`
Project name (shared, already approved): `scai`
GPU queue: `scai_q` — NVIDIA A100 80GB PCIe, CUDA 13.2, max 2 concurrent jobs/user
Program code for proxy (if not mtech, check table below): mtech = 62

## Data
Official `student_resource` zip (~1 GB), shared on Google Drive: https://drive.google.com/drive/folders/1QF5Wx8fuiap7aBsjYzTQYAU4FnDRcnuX?usp=drive_link

Download and extract it at the repo root (on padum: `~/scratch/AmazonMLChallenge`, on the login node with the proxy from step 3):
```
curl -L -o 6ab10eb3b23ba_student_resource.zip "https://drive.usercontent.google.com/download?id=1De_3Cgg3C8jN1ULdiU_UxiQYiKt6NnU7&export=download&confirm=t"
unzip -q 6ab10eb3b23ba_student_resource.zip
rm -rf __MACOSX && find student_resource -name .DS_Store -delete
```
You get `student_resource/{dataset/{train,test},utils}`, the layout the code expects. The zip and `student_resource/` are gitignored.

## 1. Get HPC account
1. Apply: https://userm.iitd.ernet.in/usermanage/hpc.html
2. Login with your Kerberos ID/password
3. Faculty supervisor (uid): `raunakbh`
4. Expiry date: 15 December
5. Wait for approval email (from noreply@hpc.iitd.ac.in, sent to you + raunakbh)
6. On first SSH login, run once:
   ```
   source /home/apps/skeleton/oneTimeHPCAccountEnvSetup.sh
   ```

## 2. Connect
```
ssh -X {kerberos_id}@hpc.iitd.ac.in
```
- This is the official CPU login host. `-X` enables X11 forwarding.
- Do NOT use `login1.hpc.iitd.ernet.in` — outdated alias, causes DNS issues off the exact IITD subnet.
- Must be on IITD campus WiFi/LAN, or IITD VPN if off-campus.
- You land on a **login node** — never run compute here directly (30 warnings = account ban).

**VSCode Remote-SSH shortcut**: add to `~/.ssh/config` on your laptop:
```
Host padum
  HostName hpc.iitd.ac.in
  User {kerberos_id}
  ForwardX11 yes
  ServerAliveInterval 60
```
Then in VSCode: green Remote icon (bottom-left) → Connect to Host → `padum`.

## 3. Internet on login node (needed for wget/pip/git)
1. Copy `proxy.sh` (in this repo) to your HPC home:
   ```
   scp proxy.sh {kerberos_id}@hpc.iitd.ac.in:~/
   ```
2. On HPC, edit the script and fill in your own username/password:
   ```
   nano proxy.sh
   ```
   **Never commit your edited copy with a real password back to git.**
3. Run:
   ```
   chmod +x proxy.sh
   ./proxy.sh &
   export http_proxy="http://proxy{PROGRAM_CODE}.iitd.ac.in:3128"
   export https_proxy="http://proxy{PROGRAM_CODE}.iitd.ac.in:3128"
   ```
   mtech → PROGRAM_CODE 62, e.g. `http://proxy62.iitd.ac.in:3128`
4. Test: `wget google.com` (then `rm index.html`)

Program codes: btech=22, dual/mtech=62, diit/integrated=21, phd=61, faculty/retfaculty=82, staff/irdstaff/mba/mdes/msc/msr/pgdip=21

## 4. Conda (Miniconda, installed to scratch — NOT home, saves quota)
```
mkdir -p ~/scratch/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p ~/scratch/miniconda3
rm ~/miniconda.sh
~/scratch/miniconda3/bin/conda init bash
source ~/.bashrc
```
Accept Anaconda ToS once (required before creating envs):
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
Create shared team env name so everyone's scripts work the same:
```
conda create -n amlc python=3.11 -y
conda activate amlc
pip install -r requirements.txt
```

## 5. Interactive GPU job (for dev/debugging)
```
tmux new -s amlc              # survives dropped SSH connections
qsub -I -P scai -q scai_q -lselect=1:ncpus=1:ngpus=1 -lwalltime=4:00:00
```
Once it starts and you land on a compute node, verify:
```
nvidia-smi
```
Should show NVIDIA A100 80GB PCIe.

Notes:
- Max walltime 168h, max 10 concurrent jobs total, max 2 on `scai_q` per user
- `exit` to end the job and free the GPU

## 6. Batch job (actual training runs, submit and walk away)
Use `batchjob.sh` as template — already set to `-P scai -q scai_q`.
Edit walltime and the python entrypoint path, then:
```
qsub batchjob.sh
qstat -T -u {kerberos_id}      # check status
qstat -ans {job_id}            # details
qdel {job_id}                  # cancel
```

## 7. Research proxy (fast dataset download, inside a job only)
```
./research_proxy.sh &
export https_proxy=http://10.10.88.6:3128
export http_proxy=http://10.10.88.6:3128
```
Only one person/session cluster-wide can use this at a time. Compute speed unaffected — only download speed differs from standard proxy.

## 8. Storage
- Home (`$HOME`): 100GB — code, conda env
- Scratch (`$HOME/scratch`): 2TB — dataset, checkpoints, conda install itself

## 9. Sharing data between team accounts
```
setfacl -m u:{TEAMMATE_KERBEROS_ID}:rwx $HOME/scratch/datasets
chmod og+rwx $HOME/scratch/datasets
```

## 10. Check compute balance
```
amgr login
amgr ls project
amgr checkbalance project -n scai
```

## Common pitfalls (hit these already, save yourself the time)
- `login1.hpc.iitd.ernet.in` doesn't resolve reliably — use `hpc.iitd.ac.in`
- wget/git hang forever on login node without proxy exported first
- Don't install full Anaconda (~1GB) — Miniconda is enough and much smaller
- Install conda to `~/scratch`, not `$HOME` — envs with torch/cuda get multi-GB
- `conda create` fails with a ToS error the first time — accept ToS once, see step 4
- No GPU visible on login node — `nvidia-smi` only works inside an allocated job

## Reference
Base steps from https://github.com/mahesh-keswani/HPC-Details
