"""scripts/lib_nav_start.sh: stop_nav_stack must kill the stack's processes but NOT its own shell (host-only test, no ROS / Docker)."""
import subprocess
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "scripts" / "lib_nav_start.sh"


class StopNavStack(unittest.TestCase):
    def test_kills_matching_processes_and_survives_itself(self):
        marker = "zzqq_stack_marker_91"
        # the caller's own command line contains the marker (like the real `bash -c '... ros2 launch ...'`): the old pkill -f killed it too
        script = f'''source {LIB}
exec -a "{marker} child" sleep 60 &
CHILD=$!
sleep 0.3
NAV_KILL_PATTERN="{marker}" NAV_STOP_GRACE=0 stop_nav_stack
sleep 0.3
kill -0 $CHILD 2>/dev/null && echo CHILD_ALIVE || echo CHILD_DEAD
echo SURVIVED'''
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.stdout.split(), ["CHILD_DEAD", "SURVIVED"], r.stderr)
        self.assertEqual(r.returncode, 0)

    def test_old_pkill_would_have_killed_the_caller(self):
        """Documents the bug: pkill -f with a pattern that appears in the caller's own command line kills the caller."""
        marker = "zzqq_stack_marker_92"
        script = f'pkill -KILL -f "{marker}" 2>/dev/null; echo SURVIVED'
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        self.assertNotIn("SURVIVED", r.stdout)
        self.assertEqual(r.returncode, -9)

    def test_script_has_bounded_controller_wait(self):
        text = LIB.read_text()
        self.assertIn("NAV_CTRL_TIMEOUT", text)
        self.assertNotIn("seq 180", text)


if __name__ == "__main__":
    unittest.main()
