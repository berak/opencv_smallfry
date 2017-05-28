kt = C:/p/kotlinc/
kc = bin\kotlinc
cp = -classpath $(kt)bin31/opencv-310.jar
lp = -Pjava.library.path=$(kt)bin31\Release
ta = ocv
all: $(ta).jar
$(ta).jar:
	$(kc) $(ta).kt $(cp) -d $(ta).jar
run: all
	bin/kotlinc -classpath "C:\p\kotlinc\bin31\opencv-310.jar;ocv.jar"
clean:
	rm $(ta).jar
