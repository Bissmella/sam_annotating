python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=172.16.36.8 --master_port=1234 \
annotate_dist.py