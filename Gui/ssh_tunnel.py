from subprocess import PIPE, Popen
from threading  import Thread

from queue import Queue, Empty

import re
import subprocess

import sys, time

ON_POSIX = 'posix' in sys.builtin_module_names

class SSHTunnel():
    def __init__(self, app):
        self.app=app
        self.job_id="xxx_xxx"

    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def poller(self,  ssh, queue, t):
        fast = 0.01
        slow = 1

        job_may_not_be_alive=True
        job_is_dead=True
        timeout = fast
        time_out_counter = 0
        while ssh.poll() is None:
            time.sleep(timeout)
            time_out_counter = (time_out_counter + 1) % 5
            if time_out_counter == 0 and not job_is_dead:
                ssh.stdin.write(f"squeue -u {self.app.short_user_id}\n")
            try:
                output = queue.get_nowait()
                timeout = fast
                found_PARTITION = not type(re.search("PARTITION", output)) is type(None)
                job_life_res= re.search(f"\s+(\d*).*{self.app.short_user_id}  R ", output)
                job_life_sign = not type(job_life_res) is type(None)

                if job_life_sign:
                    self.job_id = job_life_res.group(1)
                    self.app.textbox_active_job.setText(output)
                    job_may_not_be_alive= False
                    job_is_dead=False
                if found_PARTITION:
                    if job_may_not_be_alive:
                        job_is_dead= True
                        print(f"Job {self.job_id} has finished")
                    job_may_not_be_alive=True
                match = re.search("squeue -u", output)
                suppress = found_PARTITION or not type(match) is type(None) or job_life_sign
                if not suppress:
                    print(output, end='')
                job_id_match = re.search("Submitted batch job (\d*)", output)
                if not type(job_id_match) is type(None):
                    job_is_dead=False
                    self.job_id = job_id_match[1]
                    print("We submitted job", self.job_id)
            except Empty:
                timeout = slow
            if False:
                print("process has finished")
                ssh.terminate()
                t._stop()
                break
            # Do something else

    def open(self):
        self.ssh = Popen(['sshpass', '-p', self.app.remote_password, 'ssh', '-tt', '-o', 'StrictHostKeyChecking=no', '-A',
                     self.app.remote_user_name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1, close_fds=ON_POSIX)
        self.q = Queue()
        self.t = Thread(target=self.enqueue_output, args=(self.ssh.stdout, self.q))
        self.t.daemon = True  # thread dies with the program
        self.t.start()
        self.reader = Thread(target=self.poller, args= (self.ssh, self.q, self.t))
        self.reader.daemon = True  # thread dies with the program
        self.reader.start()
        self.ssh.stdin.write("ssh student.login \n")
        self.ssh.stdin.write("ssh mlp1 \n")
        self.ssh.stdin.write("source .bashrc \n")
        self.ssh.stdin.write("source activate mlp \n")
        self.ssh.stdin.write(f"cd {self.app.remote_scripts_path} \n")

    def start_batch(self):
        self.ssh.stdin.write(f"sbatch {self.app.script_file}\n")

    def delete_out_files(self):
        self.ssh.stdin.write(f"rm *.out\n")

    def show_out(self):
        self.ssh.stdin.write(f"echo \"$(cat slurm-{self.job_id}.out)\"\n")

    def check_active(self):
        self.ssh.stdin.write(f"squeue -u {self.app.short_user_id}\n")



