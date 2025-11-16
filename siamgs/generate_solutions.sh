#!/bin/bash

#this script will loop through all assemblies specifies in ASSEMBLY_DIR and call both the standard and interface version of the poisson solver on them.
#information from std::cout will be put into *.log.txt files and performance and time information will be put into *.memory.txt files, each in a subdirectory 
#of LOG_DIR

#print help
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
	echo "Usage: " $0 [N_START] [N_REFINE] [ASSEMBLY_DIR] [SOLUTION_DIR] [LOG_DIR]
	echo "    N_START:      the number of elements in each cardinal direction of the coarsest mesh (default: 4)"
	echo "    N_REFINE:     the number of times the interface elements are refined (default: 2)"
	echo "    ASSEMBLY_DIR: the directory containing the assemblies that specify the geometry of the problems (default: ./assemblies)"
	echo "    SOLUTION_DIR: the directory where solutions (.vtk files) will be saved to (default: ./solutions)"
	echo "    LOG_DIR:      the directory where log files will be saved (default: ./log)"
	echo "                      - std::cout logs from the programs will be stored in *.log.txt"
	echo "                      - memory information from /usr/bin/time will be stored in *.memory.txt"
	echo ""
	exit 0
fi


#config
INTERFACE_PROGRAM="interface"
STANDARD_PROGRAM="standard"

N_START=${1:-4} #starting resolution. first argument. defaults to 4.
N_REFINE=${2:-2} #number of refinements in each experiment. second argument. defualts to 2 for performance reasons. run with 4 or 5 for detailed results.
ASSEMBLY_DIR=${3:-"./assemblies"}
SOLUTION_DIR=${4:-"./solutions"}
LOG_DIR=${5:-"./logs"}

#create log directories
echo "Creating log directories"
echo "${LOG_DIR}/interface"
rm -rf "${LOG_DIR}/interface"
mkdir -p "${LOG_DIR}/interface"

echo "${LOG_DIR}/standard"
rm -rf "${LOG_DIR}/standard"
mkdir -p "${LOG_DIR}/standard"

#create solution directories
echo "Creating solution directories"
echo "${SOLUTION_DIR}/interface"
rm -rf "${SOLUTION_DIR}/interface"
mkdir -p "${SOLUTION_DIR}/interface"

echo "${SOLUTION_DIR}/standard"
rm -rf "${SOLUTION_DIR}/standard"
mkdir -p "${SOLUTION_DIR}/standard"

#compile the programs
make clean
make ${INTERFACE_PROGRAM}
make ${STANDARD_PROGRAM}


#quit if the coarse mesh is impossible to create
if ((N_START < 1 )); then
	echo "ERROR: N_START must be at least 1"
	exit 1
fi

#run both programs on each assembly
for assembly_file in "${ASSEMBLY_DIR}"/*; do
	#skip anything that isn't a file or any files that were removed
	if [ ! -f "$assembly_file" ]; then
		continue
	fi

	#get the filename without the path
	base_file=$(basename "$assembly_file" .txt)
	

	#run the standard program
	experiment_name="${base_file}.standard"
	echo "Running ${experiment_name}"
	logfile="${LOG_DIR}/standard/${experiment_name}.log.txt"
	memfile="${LOG_DIR}/standard/${experiment_name}.memory.txt"
	outdirectory="${SOLUTION_DIR}/standard/"
	/usr/bin/time -v ./${STANDARD_PROGRAM} ${N_START} ${N_REFINE} ${assembly_file} ${experiment_name} ${outdirectory} \
		> ${logfile} \
		2> ${memfile}


	#append time and system information to the log
	{
		echo ""
		echo ""
		echo "===== SYSTEM INFORMATION ====="
		echo "Date        : $(date)"
		echo "Hostname    : $(hostname)"
		echo "CPU         : $(lscpu | grep 'Model name' | cut -d ':' -f 2 | xargs)"
		echo "CPU cores   : $(nproc)"
		echo "Total RAM   : $(free -h | grep Mem | awk '{print $2}')"

		echo ""
		echo "===== EXPERIMENT PARAMETERS ====="
		echo "N_START     : ${N_START}"
		echo "N_REFINE    : ${N_REFINE}"
		echo "ASSEMBLY_DIR: ${ASSEMBLY_DIR}"
		echo "SOLUTION_DIR: ${SOLUTION_DIR}"
		echo "LOG_DIR     : ${LOG_DIR}"
	} >> ${logfile}





	#run the interface program
	experiment_name="${base_file}.interface"
	echo "Running ${experiment_name}"
	logfile="${LOG_DIR}/interface/${experiment_name}.log.txt"
	memfile="${LOG_DIR}/interface/${experiment_name}.memory.txt"
	outdirectory="${SOLUTION_DIR}/interface/"

	/usr/bin/time -v ./${INTERFACE_PROGRAM} ${N_START} ${N_REFINE} ${assembly_file} ${experiment_name} ${outdirectory} \
		> ${logfile} \
		2> ${memfile}

	#append time and system information to the log
	{
		echo ""
		echo ""
		echo "===== SYSTEM INFORMATION ====="
		echo "Date        : $(date)"
		echo "Hostname    : $(hostname)"
		echo "CPU         : $(lscpu | grep 'Model name' | cut -d ':' -f 2 | xargs)"
		echo "CPU cores   : $(nproc)"
		echo "Total RAM   : $(free -h | grep Mem | awk '{print $2}')"

		echo ""
		echo "===== EXPERIMENT PARAMETERS ====="
		echo "N_START     : ${N_START}"
		echo "N_REFINE    : ${N_REFINE}"
		echo "ASSEMBLY_DIR: ${ASSEMBLY_DIR}"
		echo "SOLUTION_DIR: ${SOLUTION_DIR}"
		echo "LOG_DIR     : ${LOG_DIR}"
	} >> ${logfile}

	

done