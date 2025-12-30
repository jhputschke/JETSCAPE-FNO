# The libraries for pythia, fastjet, and root ( "LIB_TRI" )
PY8PATH=/home/davidstewart/software/pythia8310
LIB_PYTH=-I${PY8PATH}/include -L${PY8PATH}/lib -lpythia8
LIB_FASTJET=`/home/davidstewart/software/fastjet-install/bin/fastjet-config --cxxflags --libs`
LIB_ROOT=`root-config --cflags --glibs`
LIB_TRI=${LIB_PYTH} ${LIB_FASTJET} ${LIB_ROOT} -lRecursiveTools

CC=g++
CFLAGS=-std=c++17 -O3 -Wno-deprecated

DICT_HDR = src/FlatDict_LinkDef.h
DICT_SRC = obj/FlatDict.cxx
DICT_OBJ = obj/FlatDict.o

bin/flat3d: obj/flat3d.o ${DICT_OBJ}
	${CC} ${CFLAGS} -o $@ $^ ${LIB_TRI}

${DICT_SRC}:
	rootcling -f $@ -c ${DICT_HDR}

${DICT_OBJ}: ${DICT_SRC}
	${CC} ${CFLAGS} ${LIB_ROOT} -c $< -o $@


# compilation option
obj/profiles.o: src/profiles.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@

obj/flat3d.o: src/flat3d.cc 
	${CC} ${CFLAGS} ${LIB_TRI} -c $< -o $@
