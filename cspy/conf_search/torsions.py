import pdb
import os
import sys
import math


def main():
	### Asks for and assigns the variables

	if len(sys.argv) == 1:
		ident=input("Name of structure files:")
		number=input("Number of structure files:")
		chiral=input("Do you want to check chirality?  (yes/no)")
		if chiral == 'yes':
			correct=input("Number of a file with correct stereochemistry:")
		rms4clust=input("RMS cutoff for clustering conformations:")
		max4clust=input("Single angle maximum for clustering:")
	else:
		ident=sys.argv[1]
		number=sys.argv[2]
		chiral=sys.argv[3]
		if chiral == 'yes':
			correct=sys.argv[4]
			rms4clust=sys.argv[5]
			max4clust=sys.argv[6]
		else:
			rms4clust=sys.argv[4]
			max4clust=sys.argv[5]

	### Set name of torsions file

	torfile=str(ident)+'torsions'


	### Calculates the reference inchi - for checking chirality

	if chiral == 'yes':
		os.system('babel '+'.*_'+str(correct)+'.mol2 '+ident+'correct.inchi')
		correct_file=open(ident+'correct.inchi')
		unsplit=correct_file.read()
		correct_chiral=correct_file.read().split('/')
		correct_file.close()



	### Opens input and output files

	input=open(torfile)
	output=open(ident+'.out', 'w')


	### Reads in the atom labels of the torsion angles to be calculated

	angles=[]
	for line in input:
		angles.append(line.split())


	### Calculates the torsion angles for all structure files

	m=1
	geoms=[]
	goodgeoms=[]
	chiralities={}
	molename=[0]*(int(number)+1)
	while m < int(number)+1:

		### Reads the atom positions from the .mol2 file of current conformation

		coords={}
		file=open(ident+str(m)+'.mol2')
		for line in file:
			if len(line.split()) == 9:
				coords[line.split()[0]]=line.split()[1:]
		file.close()	

		molename[m]=open(ident+str(m)+'.mol2').readlines()[1].strip().split("_")[-1]
		



		### Produces the inchi for the current conformation - checks that it is one of the enantiomers of correct molecule

		good=0
		if chiral=='yes':
			os.system('babel '+ident+str(m)+'.mol2 '+ident+str(m)+'.inchi')
			chiral_file=open(ident+str(m)+'.inchi')
			chirality=chiral_file.read().split('/')
			chiral_file.close()
			if "/b" in unsplit:
				if chirality[4] not in correct_chiral:
					good += 1
				stereo=5
			else:
				stereo=4
			if "/t" in unsplit:
				if "/m" in unsplit:
					chi = stereo +1
				else:
					chi = stereo
				if chirality[stereo] in unsplit:
					good=0
					if chirality[chi] in unsplit:
						chiralities[m]=0
					else:
						chiralities[m]=1
				else:
					good += 1


		### Calculate vectors a,b,c,d
		### And cross products e = (c-b)*(a-b) and f = (d-c)*(b-c)

		if good == 0:
			tors=[]
			tors.append(m)
			for angle in angles:
				CminusB=[]
				AminusB=[]
				DminusC=[]
				BminusC=[]
				n=1
				while n < 4:
					CminusB.append(float(coords[angle[2]][n])-float(coords[angle[1]][n]))
					AminusB.append(float(coords[angle[0]][n])-float(coords[angle[1]][n]))
					DminusC.append(float(coords[angle[3]][n])-float(coords[angle[2]][n]))
					BminusC.append(float(coords[angle[1]][n])-float(coords[angle[2]][n]))
					n=n+1
				E1 = CminusB[1]*AminusB[2] - CminusB[2]*AminusB[1]
				E2 = CminusB[2]*AminusB[0] - CminusB[0]*AminusB[2]
				E3 = CminusB[0]*AminusB[1] - CminusB[1]*AminusB[0]
				F1 = DminusC[1]*BminusC[2] - DminusC[2]*BminusC[1]
				F2 = DminusC[2]*BminusC[0] - DminusC[0]*BminusC[2]
				F3 = DminusC[0]*BminusC[1] - DminusC[1]*BminusC[0]
				magE = math.sqrt(E1**2 + E2**2 + E3**2)
				magF = math.sqrt(F1**2 + F2**2 + F3**2)
				CosTau = (E1*F1+E2*F2+E3*F3)/(magE*magF)
				if CosTau < -1.0:
					CosTau=-1.0
				if CosTau > 1.0:
					CosTau=1.0
				sign = E1*DminusC[0] + E2*DminusC[1] + E3*DminusC[2]
				if sign >= 0.0:
					Tau = 360.0*math.acos(CosTau)/(2.0*math.pi)
				else: Tau = -360.0*math.acos(CosTau)/(2.0*math.pi)
				tors.append(Tau)
			geoms.append(tors)
			goodgeoms.append(m)
		m=m+1
	m=len(goodgeoms)+1

	### Creates a dictionary to convert conformation number (m) and structure number

	geomsdict={}
	for n,geom in enumerate(goodgeoms):
		geomsdict[geom]=n+1


	### Writes the geometries to the output file 	

	for geom in geoms:
		o=1
		output.write(('%-5s')%(geom[0]))
		while o < len(angles)+1:
			output.write(('%15.5f') %(geom[o]))
			o=o+1
		output.write('\n')	

	input.close()
	output.close()
					
	############################################################################################
	#				Clusters Structures							#
	############################################################################################

	def clust(method, value, maximum):
		def rmscareful(struct):
			for geom2 in geoms[n:]:
				p=1
				ms=0
				invms=0
				while p < len(angles)+1:
					a = math.fabs(struct[p]-geom2[p])
					if float(angles[p-1][4]) == 1:
						if a < 360.0/(float(angles[p-1][4])*2):
							ms = ms + a**2
						else:
							ms = ms + (360.0 - a)**2
					elif float(angles[p-1][4]) == 2:
						if 0.0 <= a <= 90.0:
							ms = ms + a**2
						elif 90.0 < a <= 180.0:
							ms = ms + (180.0 - a)**2
						elif 180.0 < a <= 270.0:
							ms = ms + (a - 180.0)**2
						elif 270.0 < a <= 360.0:
							ms = ms + (360.0 - a)**2
					elif float(angles[p-1][4]) == 3:
						if 0.0 <= a <= 60.0:
							ms = ms + a**2
						elif 60.0 < a <= 120.0:
							ms = ms + (120.0 - a)**2
						elif 120.0 < a <= 180.0:
							ms = ms + (a - 120.0)**2
						elif 180.0 < a <= 240.0:
							ms = ms + (240.0 - a)**2
						elif 240.0 < a <= 300.0:
							ms = ms + (a - 240.0)**2
						elif 300.0 < a <= 360.0:
							ms = ms + (360.0 - a)**2

					inva = math.fabs(struct[p]+geom2[p])
					if float(angles[p-1][4]) == 1:
						if inva < 360.0/(float(angles[p-1][4])*2):
							invms = invms + inva**2
						else:
							invms = invms + (360.0 - inva)**2
					elif float(angles[p-1][4]) == 2:
						if 0.0 <= inva <= 90.0:
							invms = invms + inva**2
						elif 90.0 < inva <= 180.0:
							invms = invms + (180.0 - inva)**2
						elif 180.0 < inva <= 270.0:
							invms = invms + (inva - 180.0)**2
						elif 270.0 < inva <= 360.0:
							invms = invms + (360.0 - inva)**2
					elif float(angles[p-1][4]) == 3:
						if 0.0 <= inva <= 60.0:
							invms = invms + inva**2
						elif 60.0 < inva <= 120.0:
							invms = invms + (120.0 - inva)**2
						elif 120.0 < inva <= 180.0:
							invms = invms + (inva - 120.0)**2
						elif 180.0 < inva <= 240.0:
							invms = invms + (240.0 - inva)**2
						elif 240.0 < inva <= 300.0:
							invms = invms + (inva - 240.0)**2
						elif 300.0 < inva <= 360.0:
							invms = invms + (360.0 - inva)**2
					p=p+1
				rms=math.sqrt(ms/float(len(angles)))
				invrms=math.sqrt(invms/float(len(angles)))
				rmses.append([struct[0], geom2[0], [rms, invrms]])
				carefulrmsesfile.write(('%-5s%-5s%15.5f%15.5f')%(struct[0], geom2[0], rms, invrms)+ '\n')

				p=1
				dtor=[]
				invdtor=[]
				dtortemp=0
				invdtortemp=0
				while p < len(angles)+1:
					dtor.append(p)
					dtortemp = math.fabs(struct[p]-geom2[p])
					if float(angles[p-1][4]) == 1:
						if dtortemp < 180.0:
							dtor[p-1]=dtortemp
						else:
							dtor[p-1]= 360.0 - dtortemp
					elif float(angles[p-1][4]) == 2:
						if 0.0 <= dtortemp <= 90.0:
							dtor[p-1]=dtortemp
						elif 90.0 < dtortemp <= 180.0:
							dtor[p-1] = 180.0 - dtortemp
						elif 180.0 < dtortemp <= 270.0:
							dtor[p-1] = dtor[p-1] - 180.0
						elif 270.0 < dtortemp <= 360.0:
							dtor[p-1] = 360.0 - dtortemp
					elif float(angles[p-1][4]) == 3:
						if 0.0 <= dtortemp <= 60.0:
							dtor[p-1]=dtortemp
						elif 60.0 < dtortemp <= 120.0:
							dtor[p-1] = 120.0 - dtortemp
						elif 120.0 < dtortemp <= 180.0:
							dtor[p-1] = dtortemp - 120.0
						elif 180.0 < dtortemp <= 240.0:
							dtor[p-1] = 240.0 - dtortemp
						elif 240.0 < dtortemp <= 300.0:
							dtor[p-1] = dtortemp - 240.0
						elif 300.0 < dtortemp <= 360.0:
							dtor[p-1] = 360.0 - dtortemp
					invdtor.append(p)
					invdtortemp = math.fabs(struct[p]+geom2[p])
					if float(angles[p-1][4]) == 1:
						if invdtortemp < 180.0:
							invdtor[p-1]=invdtortemp
						else:
							invdtor[p-1]= 360.0-invdtortemp
					elif float(angles[p-1][4]) == 2:
						if 0.0 <= invdtortemp <= 90.0:
							invdtor[p-1]=invdtortemp
						elif 90.0 < invdtortemp <= 180.0:
							invdtor[p-1] = 180.0 - invdtortemp
						elif 180.0 < invdtortemp <= 270.0:
							invdtor[p-1] = invdtortemp - 180.0
						elif 270.0 < invdtortemp <= 360.0:
							invdtor[p-1] = 360.0 - invdtortemp
					elif float(angles[p-1][4]) == 3:
						if 0.0 <= invdtortemp <= 60.0:
							invdtor[p-1]=invdtortemp
						elif 60.0 < invdtortemp <= 120.0:
							invdtor[p-1] = 120.0 - invdtortemp
						elif 120.0 < invdtortemp <= 180.0:
							invdtor[p-1] = invdtortemp - 120.0
						elif 180.0 < invdtortemp <= 240.0:
							invdtor[p-1] = 240.0 - invdtortemp
						elif 240.0 < invdtortemp <= 300.0:
							invdtor[p-1] = invdtortemp - 240.0
						elif 300.0 < invdtortemp <= 360.0:
							invdtor[p-1] = 360.0 - invdtortemp
					p=p+1
				maxtors.append((struct[0], geom2[0], [max(dtor), max(invdtor)]))
				carefulmaxtorsfile.write(('%-5s%-5s%15.5f%15.5f')%(struct[0], geom2[0], max(dtor), max(invdtor))+ '\n')

		clusters=[]
		clusters.append([])
		clusters[0].append(0)
		clusters[0].append([])	
		for i in range(m-1):
			clusters.append([])
			clusters[i+1].append(goodgeoms[i])
			clusters[i+1].append([])

		if method == 'careful':
			clusterfile=open('clusters_'+method+'_'+str(value)+'_'+str(maximum)+'.txt', 'w')
			clustered=[]
			rmses=[]
			maxtors=[]
			carefulrmsesfile=open('rmsout.txt', 'w')
			carefulmaxtorsfile=open('maxdtor.txt', 'w')
			for n,geom in enumerate(geoms):
				rmscareful(geom)
			carefulrmsesfile.close()
			rmses2=rmses[:]
			for n,rms in enumerate(rmses):
				if rms[0] == rms[1] and rms[0] not in clustered:
					clusters[geomsdict[rms[0]]][1].append([rms[1],0,molename[rms[1]]])
				else:
					if rms[0] not in clustered:
						if chiral == 'yes':
							chirality1=chiralities[rms[0]]
							chirality2=chiralities[rms[1]]
							if chirality1 == chirality2:
								if maxtors[n][2][0] < float(maximum) and rms[2][0] < float(value):
									clustered.append(rms[1])
									clusters[geomsdict[rms[0]]][1].append([rms[1],0,molename[rms[1]]])

							else:
								if maxtors[n][2][1] < float(maximum) and rms[2][1] < float(value):
									clustered.append(rms[1])
									clusters[geomsdict[rms[0]]][1].append([rms[1],1,molename[rms[1]]])

						else:
							if maxtors[n][2][0] < float(maximum) and rms[2][0] < float(value):
								clustered.append(rms[1])
								clusters[geomsdict[rms[0]]][1].append([rms[1],0,molename[rms[1]]])

							else:
								if maxtors[n][2][1] < float(maximum) and rms[2][1] < float(value):
									clustered.append(rms[1])
									clusters[geomsdict[rms[0]]][1].append([rms[1],1,molename[rms[1]]])


		maxlen=0		
		for cluster in clusters[1:]:
			if len(cluster[1]) == 0:
				clusters.remove(cluster)
			elif len(cluster[1]) >= maxlen:
				maxlen = len(cluster[1])

		if method=='careful':
			for cluster in clusters[1:]:
				if len(cluster[1])==1:
					rmsa=0.0
				else:
					rmsa=0.0
					for thing in cluster[1][1:]:
						for n,thing2 in enumerate(rmses):
							if thing2[1]==thing[0] and thing2[0]==cluster[0]:
								rmsa=rmsa+thing2[2][thing[1]]
					rmsa=rmsa/(len(cluster[1])-1)
				cluster.append(rmsa)

		for cluster in clusters[1:]:
			clusterfile.write(('%-5s')%(cluster[0]))
			if method == 'rms' or method =='careful':
				clusterfile.write(('%10.5f')%(cluster[2])+'\t')
			for thing2 in cluster[1]:
				clusterfile.write(('%-4s')%(thing2))
			clusterfile.write('\n')
		clusterfile.close()
		return [rmses, maxtors, rmses2, clusters, clustered]
	clustout=clust('careful',rms4clust,max4clust)
	rmses=clustout[0]
	maxtors=clustout[1]
	rmses2=clustout[2]
	clusters=clustout[3]
	clustered=clustout[4]

	############################################################################################
	#			Reclusters Structures using calculated RMS and Max values		#
	############################################################################################

	#print('You may now recluster the results')
	#print('To recluster structures enter the command reclust(method, precision). Method can be \'rms\' or \'max\', precision is the cutoff in degrees.')


	def reclust(method, value, maximum=120.0):

		clusters=[]
		clusters.append([])
		clusters[0].append(0)
		clusters[0].append([])	
		for i in range(m-1):
			clusters.append([])
			clusters[i+1].append(goodgeoms[i])
			clusters[i+1].append([])
		if method == 'careful':
			clusterfile=open('clusters_'+method+'_'+str(value)+'_'+str(maximum)+'.txt', 'w')
			rmses2=rmses[:]
			for n,rms in enumerate(rmses):
				if type(rmses2[n])!=str:
					if maxtors[n][2][0] < maximum and rms[2][0] < float(value):
						clusters[geomsdict[rms[0]]][1].append([molename[rms[1]],0])
						if rms[0] != rms[1]:
							for p in range( (m-geomsdict[rms[1]]) ):
								rmses2[int((float(geomsdict[rms[1]]-1)/2)*(2*(m-1)-(float(geomsdict[rms[1]]-1)-1))+p)]='pass'
					elif maxtors[n][2][1] < maximum and rms[2][1] < float(value):
						clusters[geomsdict[rms[0]]][1].append([molename[rms[1]],1])
						if rms[0] != rms[1]:
							for p in range( (m-geomsdict[rms[1]]) ):
								rmses2[int((float(geomsdict[rms[1]]-1)/2)*(2*(m-1)-(float(geomsdict[rms[1]]-1)-1))+p)]='pass'
		
		else:
			print('Initial clustering for chosen recluster method has not been done. Please choose another method or perform initial clustering using clust(method, value, maximum)')
		
		maxlen=0
		for cluster in clusters[1:]:
			if len(cluster[1]) == 0:
				clusters.remove(cluster)
			elif len(cluster[1]) >= maxlen:
				maxlen = len(cluster[1])

		if method == 'rms' or method=='careful':
			for cluster in clusters[1:]:
				if len(cluster[1])==1:
					rmsa=0.0
				else:
					rmsa=0
					for thing in cluster[1][1:]:
						for n,thing2 in enumerate(rmses):
							if thing2[1]==thing[0] and thing2[0]==cluster[0]:
								rmsa=rmsa+thing2[2][thing[1]]
					rmsa=rmsa/(len(cluster[1])-1)
				cluster.append(rmsa)

		for cluster in clusters[1:]:
			clusterfile.write(('%-5s')%(cluster[0]))
			if method == 'rms' or method=='careful':
				clusterfile.write(('%10.5f')%(cluster[2])+'\t')
			for thing2 in cluster[1]:
				clusterfile.write(('%-4s')%(thing2))
			clusterfile.write('\n')
		clusterfile.close()	
				


	############################################################################################
	#			Produces RMS matrices for the clusters					#
	############################################################################################



	def matrix():
		matrix=[]
		for i in range(m):
			n=i
			matrix.append([])
			while n < m+i:
				matrix[i].append(n)
				n=n+1

		for rms in rmses:
			matrix[geomsdict[rms[0]]][geomsdict[rms[1]]]=rms[2]
			matrix[geomsdict[rms[1]]][geomsdict[rms[0]]]=rms[2]

		matrices=[]
		for n,cluster in enumerate(clusters[1:]):
			clustmat=[]
			for i in range(len(cluster[1])):
				j=i
				clustmat.append([])
				while j < len(cluster[1]):
	#				clustmat[i].append(matrix[geomsdict[cluster[1][i][0]]][geomsdict[cluster[1][j][0]]][cluster[1][j][1]])
					clustmat[i].append(min(matrix[geomsdict[cluster[1][i][0]]][geomsdict[cluster[1][j][0]]]))
					j=j+1

			clustmat.insert(0, cluster[1])
			matrices.append(clustmat)	

		mat=open('matrices_'+str(rms4clust)+'_'+str(max4clust), 'w')
		for j,grid in enumerate(matrices):
			mat.write(('%-10s\t')%(j+1))
			for element in grid[0]:
				mat.write(('%10s\t')%(element))
			mat.write('\n')
			for n,row in enumerate(grid[1:]):
				mat.write(('%10s\t')%(grid[0][n]))
				j=0
				while j < n:
					mat.write(('%10s\t')%(' '))
					j=j+1
				for element in row:
					mat.write(('%10.5f\t')%(element))
				mat.write('\n')
			mat.write('\n')
		mat.close()

		return(matrix, matrices)

	jim=matrix()

if __name__ == "__main__":
	main()