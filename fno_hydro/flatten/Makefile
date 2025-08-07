LIB_ROOT=`root-config --cflags --glibs` -lRecursiveTools

CC=g++
CFLAGS=-std=c++17 -O3 -Wno-deprecated

DICT_HDR = src/FlatDict_LinkDef.h
DICT_SRC = obj/FlatDict.cxx
DICT_OBJ = obj/FlatDict.o

# Create directories if they don't exist
bin/flat: obj/flat.o ${DICT_OBJ} | bin
	${CC} ${CFLAGS} -o $@ $^ ${LIB_ROOT}
	
.PHONY: directories
directories: bin obj

bin:
	mkdir -p bin

obj:
	mkdir -p obj

bin/stats: obj/stats.o ${DICT_OBJ} | bin
	${CC} ${CFLAGS} -o $@ $^ ${LIB_ROOT}

${DICT_SRC}: | obj
	rootcling -f $@ -c ${DICT_HDR}

${DICT_OBJ}: ${DICT_SRC} | obj
	${CC} ${CFLAGS} ${LIB_ROOT} -c $< -o $@


# compilation option

obj/stats.o: src/stats.cc | obj
	${CC} ${CFLAGS} ${LIB_ROOT} -c $< -o $@

obj/flat.o: src/flat.cc | obj
	${CC} ${CFLAGS} ${LIB_ROOT} -c $< -o $@
