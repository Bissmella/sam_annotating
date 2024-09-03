python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr=172.16.79.1 --master_port=1234 \
annotate_dist.py