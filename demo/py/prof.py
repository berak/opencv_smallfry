
f0 = open("p0.txt","r")
f1 = open("p1.txt","r")

def line(f):
	l0 = f.readline()
	if not l0: return []
	#print l0
	ss = l0.strip().split(" ")
	zz = []
	for s in ss:
		if s == "": continue
		zz.append(s)
	return zz

names = []
counts = []
t = [[],[]]
while True:
	z0 = line(f0)
	z1 = line(f1)
	if len(z0)<1: break
	counts.append(int(z0[1]))
	t[0].append(float(z0[2]))
	t[1].append(float(z1[2]))
	names.append(z0[0])
print "%-20s %-9s  %-9s %-9s %-9s" % ("function", "counts", "ocl_off", "ocl_on", "ratio")
for i in range(len(names)):
	if t[0][i]!=0:
		r = t[1][i] /  t[0][i]
	else:
		r = t[1][i]
	print "%-20s %-9d  %4.6f  %4.6f  %4.6f" % (names[i], counts[i], t[0][i], t[1][i], r)