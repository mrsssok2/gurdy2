# Manual-goal velocity diagnostic (run in the sim container, nothing else running): scripts/nav_velocity_diag.sh ground_truth|ekf
# Sends 3 goals from rest and prints the largest MPPI (cmd_vel_nav) and min-speed-node outputs. Found the missing controller_server odom_topic patch.
# usage: diag.sh ODOM  -> sim up, then manual goals east/north/west from rest, prints MPPI/smoothed/lifted commands
ODOM=$1
cd ~/gurdy2; set +u; source ros2_ws/install/setup.bash
export PYTHONPATH=$HOME/gurdy2/gurdy_core:$PYTHONPATH
mkdir -p /tmp/o; source scripts/lib_nav_start.sh
start_nav_stack /tmp/o gui:=false nav:=slam navigation:=true controller:=policy odom_source:=$ODOM || exit 1
sleep 10
run() { # x y yaw label
  (timeout 22 ros2 topic echo /cmd_vel_nav --csv > /tmp/cmd_$4.csv 2>&1 &)
  (timeout 22 ros2 topic echo /cmd_vel_min --csv > /tmp/min_$4.csv 2>&1 &)
  (timeout 22 ros2 topic echo /odom/ground_truth --csv --field pose.pose.position > /tmp/gt_$4.csv 2>&1 &)
  (timeout 22 scripts/nav_goal.sh $1 $2 $3 > /tmp/g_$4.txt 2>&1 &)
  sleep 24
  python3 - <<PY
import numpy as np
def load(f,n):
    a=[l.strip().split(',') for l in open(f) if l.count(',')>=5]
    return np.array([[float(x) for x in r[:n]] for r in a]) if a else np.zeros((1,n))
v=load('/tmp/cmd_$4.csv',6); m=load('/tmp/min_$4.csv',6); g=load('/tmp/gt_$4.csv',3)
print('$4: nav max|vx| %.3f |vy| %.3f |wz| %.3f ; min-speed out max|vx| %.3f |vy| %.3f ; GT end x,y = %.2f %.2f'%(abs(v[:,0]).max(),abs(v[:,1]).max(),abs(v[:,5]).max(),abs(m[:,0]).max(),abs(m[:,1]).max(), g[-1][0], g[-1][1]))
PY
}
run 0.0 0.9 1.57 north
run 1.0 0.9 0.0 east
run 1.0 0.0 -1.57 south
stop_nav_stack
