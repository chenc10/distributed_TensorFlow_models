if [ -n $exp_home ]; then
		export exp_home=`pwd`
fi
if [ "$1" == "" ]; then
	echo "please specify model and sync mode (bsp, asp, etc.)!"
	exit
fi

model=$1
mode=$2

ps_hosts=''
for i in $ps
do
	ps_hosts=$ps_hosts$i":22222,"
done
ps_hosts=${ps_hosts%%,}
echo 'ps_hosts: '$ps_hosts

worker_hosts=''
for i in $workers
do
        worker_hosts=$worker_hosts$i":22223,"
done
worker_hosts=${worker_hosts%%,}
echo 'worker_hosts: '$worker_hosts

# path for training inception model (upon imagenet dataset) is different with others (upon cifar10 dataset)
if [ "$1" == "inception" ]; then
	num=0
	for i in $ps
	do
		nohup ssh $i "export CUDA_VISIBLE_DEVICES= && python /home/ubuntu/chen-models/inception/imagenet_inception_${mode}.py --job_name=ps --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" >/home/ubuntu/ps_${num}.log 2>&1 &
		num=$((num+1))
	done
	
	num=0
	for i in $workers
	do
		nohup ssh $i "python /home/ubuntu/chen-models/inception/imagenet_inception_${mode}.py  --job_name=worker --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" >/home/ubuntu/worker_$num.log 2>&1 &
		num=$((num+1))
	done
	exit
fi

num=0
for i in $ps
do
	nohup ssh $i "export CUDA_VISIBLE_DEVICES= && python $exp_home/${model}/cifar10_${model}_${mode}.py  --job_name=ps --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" >$exp_home/../ps_${num}.log 2>&1 &
	num=$((num+1))
done

num=0
for i in $workers
do
	nohup ssh $i "python $exp_home/${model}/cifar10_${model}_${mode}.py  --job_name=worker --task_id=$num --ps_hosts=$ps_hosts --worker_hosts=$worker_hosts" >$exp_home/../worker_$num.log 2>&1 &
	if [ $num -eq 0 ]; then
		nohup ssh $i "sleep 20 && export CUDA_VISIBLE_DEVICES= && python $exp_home/$model/cifar10_${model}_eval.py" >$exp_home/../model_eval.log 2>&1 &
	fi
	num=$((num+1))
done
