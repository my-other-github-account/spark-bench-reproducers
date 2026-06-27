#!/usr/bin/env bash
# run_4node.sh — orchestrate a TP=4 launch across 4 DGX Spark nodes from a control host.
#
# Edit the NODES array below for your fleet. Each entry: "ssh_target:rank:host_ip".
#   - ssh_target : how this control host reaches the node over ssh
#   - rank       : 0..3 (rank 0 is the head / rendezvous master)
#   - host_ip    : the IP each rank advertises to the others (use your fastest fabric, e.g. 100GbE)
#
# The model directory ($MODEL_DIR) must be present on every node at the same path (NFS export or a
# local copy). It is mounted read-only into the container at /model.
#
# Launch is PARALLEL (workers first, head +2s) to collapse rendezvous skew — sequential launch trips
# the 600s TCP-store DistStoreError.
set -u

IMAGE="${IMAGE:-glm52-spark-decode:latest}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/GLM-5.2-NVFP4}"
MASTER_PORT="${MASTER_PORT:-29588}"
SSH="ssh -o ConnectTimeout=12 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

# rank0 must be first. host_ip = the address each rank advertises (your fastest fabric).
NODES=(
  "node0:0:SPARK_A"
  "node1:1:SPARK_C"
  "node2:2:SPARK_E"
  "node3:3:SPARK_F"
)
MASTER_ADDR="${MASTER_ADDR:-$(echo "${NODES[0]}" | cut -d: -f3)}"

launch_rank() {
  local target="$1" rank="$2" hostip="$3"
  $SSH "$target" "
    set -e
    bash /repro/scripts/purge_node.sh
    docker rm -f vllm_node 2>/dev/null || true
    docker run -d --name vllm_node --runtime=nvidia --gpus all --privileged --ipc=host --network=host \
      -v '$MODEL_DIR:/model:ro' \
      -v \"\$HOME/.triton:/root/.triton\" \
      -e NODE_RANK=$rank -e MASTER_ADDR=$MASTER_ADDR -e HOST_IP=$hostip \
      -e MASTER_PORT=$MASTER_PORT -e NNODES=4 \
      '$IMAGE'
    echo \"launched rank=$rank on $target (host_ip=$hostip)\"
  "
}

# workers (rank>0) first, in parallel
for spec in "${NODES[@]}"; do
  IFS=: read -r target rank hostip <<< "$spec"
  [ "$rank" = "0" ] && continue
  launch_rank "$target" "$rank" "$hostip" &
done
sleep 2
# head (rank 0) last
for spec in "${NODES[@]}"; do
  IFS=: read -r target rank hostip <<< "$spec"
  [ "$rank" = "0" ] || continue
  launch_rank "$target" "$rank" "$hostip" &
done
wait
echo "all 4 ranks launched on port $MASTER_PORT."
echo "load takes ~15-20 min (47 shards). Watch: docker exec vllm_node tail -f /tmp/vllm_serve.log on rank0."
echo "When :8000 binds, run scripts/bench.sh against the head node."
