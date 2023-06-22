
To submit a job the user needs to specify the resources the job needs to be allocated to it. This is done in a bash script. I've provided an example (slurmTemplate20.sh)
of such a script. I've also provided two demo python programs (xavier.py and nuno.py) this job script calls. Its only purpose is to provide some dummy load to the CPU cores and GPUs
to demonstrate how SLURM can be used. The files slurmTemplate20.sh and xavier.py can be found on ese-hivemind in this folder (/raid/hivemind).

If you would like to run this example jobscript (slurmTemplate20.sh), you'll need to install miniconda (with default options); create an environment called "testgpu" and then
from within the "testgpu" environment:
`conda install cudatoolkit`
`conda install numba`

Copy the example files into a folder to your home directory:
Create a the "slurmEG", copy the ecamples files over, and navigate to the folder:
`mkdir ~/slurmEG`
`cp -r /raid/hivemind ~/slurmEG/.`
`cd ~/slurmEG`

To submit the jobscript:
`sbatch slurmTemplate20.sh`

Check if the job is running (R) or in pending (PD) state (ST) with:
`squeue`

A parameter specification in a SLURM job script starts with:
#SLLURM ...
if there are two hashes the it's been commented out, eg:
##SLURM ...

The std output of the job will be piped to a text file called "slurm-<jobid>.out" and the std err of the job will be piped to a text file called "slurm-<jobid>.err".

You can cancel a job with:
scancel <jobID>

You can cancel all your jobs using:
scancel -u <username>

If you are curious how long a completed job ran for or how much memory it used, you can look that information with sacct. For example:
    sacct -S `date --date "last month" +%Y-%m-%d` -o "Submit,JobID,JobName,Partition,NCPUS,State,ExitCode,Elapsed,CPUTime,MaxRSS" This will pull some basic stats for all the jobs you ran in the past month.
    sacct -l -j <job_id> will dump all the information it has about a particular job
    sacct -l -P -j <job_id> | awk -F\| 'FNR==1 { for (i=1; i<=NF; i++) header[i] = $i; next } { for (i=1; i<=NF; i++) print header[i] ": " $i; print "-----"}' same as above but expands columns to rows

Please refer to the SLURM documentation https://slurm.schedmd.com/documentation.html for further information.

Also, to view the node utilisation you can navigate to http://http://ese-hivemind.ese.ic.ac.uk:3000/
On the leftmost menu hover over the four squares and click on "Dashboards".
Select General and GPU Nodes.

Please consider the system partition in which your home folder resides as temporary. On occasion, when necessary, a clean build of the system which will delete all
environments and data that is stored there. The "/raid/" partition that should persist these infrequent measures. If a data folder for your username
does not exist on this partition yet, please get in touch and we will provide you with such a partition. Backups of your data is your responsibility.
We recommend the Research Data Store (RDS) https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/service-offering/rds/.

If you have any further questions, please don't hesitate to get in touch.

Francois van Schalkwyk
