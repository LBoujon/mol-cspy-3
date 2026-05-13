import cspy
import sys
from multiprocessing import Pool
import os

    
def overlay(file_list):
    for file in file_list:
        m2 = cspy.Molecule.load(file)
        #print("{} loaded, performing overlay".format(file))
        m3, order, rmsd = m1.overlay(m2)
        #print("overlay complete, saving to overlayed")
        m3.save("overlayed/"+str(file))

def overlay_mp(file_list):
    chunks = [file_list[i::core_count] for i in range(core_count)]
    pool = Pool(processes=core_count)
    result = pool.map(overlay, chunks)

def main():
    
    #overlay all atoms
    FILE_NAME = sys.argv[1] + '_'
    no_files = sys.argv[2]
    core_count = int(sys.argv[3])

    if not os.path.isdir('overlayed'):
        os.mkdir('overlayed')

    master = str(FILE_NAME) + "1.xyz"
    file_list = []
    for i in range(1, int(no_files)):
        i+=1
        file_list.append('{}'.format(FILE_NAME)+'{}'.format(i)+'.xyz')

    m1 = cspy.Molecule.load(master)

    overlay_mp(file_list)
    for conformer in file_list:
        os.rename("overlayed/"+str(conformer), str(conformer))
    os.rmdir("overlayed")

if __name__ == "__main__":
    main()
