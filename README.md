# Amazon ML Challenge 2026

Cluster setup for IIT Delhi PADUM (account, proxy, conda, GPU jobs) is in [HPC_SETUP.md](HPC_SETUP.md).
The pipeline, results so far and how to reproduce the best submission are in [ber/README.md](ber/README.md)
(branch `ber-pipeline`).

## Repo setup
The repo is private, so you need collaborator access.

On your PC:
```
git clone https://github.com/i-sahajmistry/Amazon-ML-Challenge-26.git AmazonMLChallenge
```

On padum, clone into `~/scratch/AmazonMLChallenge`, the path the code expects. The login node has no direct internet, so GitHub is reached over SSH on port 443 through the IITD proxy. Add this to `~/.ssh/config` on padum, using your own proxy number from [HPC_SETUP.md](HPC_SETUP.md) step 3:
```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    ProxyCommand ncat --proxy proxy62.iitd.ac.in:3128 --proxy-type http %h %p
```
Add your padum public key (`cat ~/.ssh/id_rsa.pub`) to your GitHub account, check it with `ssh -T git@github.com`, then:
```
git clone git@github.com:i-sahajmistry/Amazon-ML-Challenge-26.git ~/scratch/AmazonMLChallenge
```

## Data
Official `student_resource` zip (~1 GB), shared on Google Drive: https://drive.google.com/drive/folders/1QF5Wx8fuiap7aBsjYzTQYAU4FnDRcnuX?usp=drive_link

Download and extract it at the repo root. On padum, do this on the login node with the proxy exported. On Windows, use Git Bash.
```
curl -L -o 6ab10eb3b23ba_student_resource.zip "https://drive.usercontent.google.com/download?id=1De_3Cgg3C8jN1ULdiU_UxiQYiKt6NnU7&export=download&confirm=t"
unzip -q 6ab10eb3b23ba_student_resource.zip
rm -rf __MACOSX && find student_resource -name .DS_Store -delete
```
You get `student_resource/{dataset/{train,test},utils}`, the layout the code expects. The zip and `student_resource/` are gitignored.
