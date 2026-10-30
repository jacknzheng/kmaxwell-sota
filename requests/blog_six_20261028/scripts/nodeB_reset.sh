#!/bin/bash
# Clean teardown of all Node B lrhalf processes + partial state, then one fresh restart.
for pat in run_nodeB.sh run_arm.py bounded_train torch.distributed.run continuous20_train; do pkill -9 -f "$pat"; done
sleep 6
echo "after-kill: bounded=$(pgrep -c -f bounded_train) arm=$(pgrep -c -f run_arm) node=$(pgrep -c -f run_nodeB)"
rm -rf /root/blog_six/work/lrhalf
rm -f /root/nodeB.rc /root/blog_six/code/replay/.blog_gpu.lock /root/blog_six/work/.blog_gpu.lock
echo "work/lrhalf: $([ -d /root/blog_six/work/lrhalf ] && echo EXISTS || echo GONE)"
echo "gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
sleep 3
cd /root/blog_six && nohup bash /root/run_nodeB.sh > /root/nodeB.log 2>&1 &
echo "restart PID $!"
echo RESET_DONE
