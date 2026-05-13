#!/bin/sh 
#
if test "x${APPLICATION_BASE}" = "x" ; then
   echo "Fatal error: $0 Environment variable APPLICATION_BASE not set by calling script"
fi
#
CONQUEST_MAIN=
#
debug=0
#
test "$1" = "-d" && debug=1

test ${debug} = 1 && echo "CSDHOME       : ${CSDHOME}"

if test "x${CQLOCAL}" = "x" ; then

   BINDIR=`dirname $0`
   test ${debug} = 1 && echo "BINDIR        : ${BINDIR}"

   HERE=`pwd`
   cd ${BINDIR}/.. ; CQLOCAL=`pwd` ; cd ${HERE}

fi

if test ${debug} = 1 ; then
   if test -h $0 ; then
      echo "Symbolic Link : $0"
   fi
   echo "CQLOCAL       : ${CQLOCAL}"
   exit 0
fi

p_echo=echo
test "${mach}" = "linux" && p_echo="echo -e"
: ${echo=${p_echo}}

noexec=0

if test "x$CSDMACHINE" != "x" ; then
   mach=$CSDMACHINE
elif test -f ${CSDHOME}/bin/csdmach.sh ; then
   mach=`${CSDHOME}/bin/csdmach.sh`
elif test -f ${CQLOCAL}/bin/csdmach.sh ; then
   mach=`${CQLOCAL}/bin/csdmach.sh`
else
   echo "Could not determine machine type" >&2 
   exit 1
fi

# ensure netscape wrapper script is in the user's PATH
if test -f ${CSDHOME}/bin/nss.sh ; then
   PATH="${CSDHOME}/bin:${PATH}"
elif test -f ${CQLOCAL}/bin/nss.sh ; then
   PATH="${CQLOCAL}/bin:${PATH}"
else
   echo "Could not find netscape startup script nss.sh"
   PATH="${PATH}"
fi
export PATH

if test ! -d ${CQLOCAL}/c_${mach} ; then
   if test "x${CONQUEST_MAIN}" = "x" ; then
      noexec=1
   else
      if test ! -d ${CONQUEST_MAIN}/c_${mach} ; then
         noexec=1
      else
         CQROOT=${CONQUEST_MAIN}
      fi
   fi
else
   CQROOT=${CQLOCAL}
fi


if test ${noexec} = 1 ; then
   ${echo} "Could not find files for ${mach}"
#   exit
fi


mesa=0
mesamethod=
verify=0
cmdopt=

while  test $# -ne 0  ; do
  case "$1" in
    -mesa)
        mesa=1
        mesamethod='-mesa'
        ;;
    -verify)
        verify=1
        ;;
    -debug)
        debug=1
    ;;
#pass other options through to executable
    *)  
        cmdopt="$cmdopt $1"
        ;;

  esac

  shift

done

#ensure executable exists

if test -f ${CQROOT}/c_${mach}/bin/${APPLICATION_BASE}.x; then 

#verify mode

    if test "$verify" = "1" ; then
        echo "Found executable ${CQROOT}/c_${mach}/bin/${APPLICATION_BASE}.x"
        exit 0
    fi
    
else

    echo "Cannot find ${APPLICATION_BASE}.x executable for architecture ${mach} in ${CQROOT}/c_${mach}/bin"

fi


if test ${mesa} != 1 ; then
   if test "X${CQGRAPHICS}" = "Xmesa" ; then 
      mesa=1
      mesamethod='CQGRAPHICS'
   fi
fi

TCL_LIBRARY=${CQROOT}/share/lib/tcl ; export TCL_LIBRARY
PYTHONHOME=${CQROOT}/share:${CQROOT}/c_${mach}; export PYTHONHOME

if test "x$CCDC" = "x" ; then
    CCDC=CCDC
fi
CQLIB=${CQROOT}/c_${mach}/lib:${CQROOT}/c_${mach}/lib/qt:${CQROOT}/c_${mach}/lib/${CCDC}
GLLIB=${CQROOT}/c_${mach}/lib/MesaGL
GCCLIB=${CQROOT}/c_${mach}/lib/GCC
CCLIB=${CQROOT}/c_${mach}/lib/CC

# libexpat for SILVER
if [ -d "${CQROOT}/c_${mach}/lib/expat" ]; then
    CQLIB="${CQLIB}:${CQROOT}/c_${mach}/lib/expat"
fi

# Use this to detect OpenGL when running from SG or linux
: ${glxinfo=/usr/sbin/glxinfo}

case ${mach} in 

 
   linux)
   # need to check that a suitable glibc is installed.
     

/lib/i386-linux-gnu/libc.so.6
    if test -d  ; then
        glibc_version=`ls -l /lib/i386-linux-gnu/libc.so.6  | sed 's/^.*libc-//'  | sed 's/.so$//'`
#       echo "Version glibc $glibc_version from symbolic link"
        case "$glibc_version" in
            1.*|2.0*|2.1.[012]*)
                echo "Exiting as ${APPLICATION_BASE} requires glibc 2.1.3 or later, $glibc_version installed" >&2 
                exit 1
            ;;
            2.1.[3456789]* |2.[23456789]* )
                echo "Running ${APPLICATION_BASE} with glibc version $glibc_version"
            ;;
        esac

    else 
        if test -f /lib/i386-linux-gnu/libc.so.6; then 
        if test -f /bin/rpm ; then
# try libc
            glibc_version=`/bin/rpm -q libc 2>/dev/null | sed 's/^.*libc-//' | sed 's/-.*$//'` 
            case "$glibc_version" in
                1.*|2.0*|2.1.[012]*)
                    echo "Exiting as ${APPLICATION_BASE} requires glibc 2.1.3 or later, $glibc_version installed" >&2       
                    exit 1
                    ;;
                2.1.[3456789]* |2.[23456789]* )
                    echo "Running ${APPLICATION_BASE} with glibc version $glibc_version" 
                    ;;
            esac
        
# try glibc
            glibc_version=`/bin/rpm -q glibc 2>/dev/null | sed 's/^.*libc-//' | sed 's/-.*$//'` 
            case "$glibc_version" in
                1.*|2.0*|2.1.[012]*)
                    echo "Exiting as ${APPLICATION_BASE} requires glibc 2.1.3 or later, $glibc_version installed" >&2       
                    exit 1
                    ;;
                2.1.[3456789]* |2.[23456789]* )
                    echo "Running ${APPLICATION_BASE} with glibc version $glibc_version" 
                    ;;
            esac            
        fi      
        
        else
        echo "Warning: ${APPLICATION_BASE} requires /lib/i386-linux-gnu/libc.so.6 but not found"
        fi  
    fi
   
         CQLIB=${CQLIB}:${GCCLIB}:${CQ_LD_LIBRARY_PATH}
         if test "X${LD_LIBRARY_PATH}" != "X" ; then
              LD_LIBRARY_PATH=${CQLIB}:${LD_LIBRARY_PATH}
         else
              LD_LIBRARY_PATH=${CQLIB}
         fi
         export LD_LIBRARY_PATH         ;;   
     
   # rs6000 requires  LIBPATH ; setting LD_LIBRARY_PATH as well for safety

   rs6000)  
         if test "X${LIBPATH}" != "X" ; then
              LIBPATH=${CQLIB}:${LIBPATH}
         else
              LIBPATH=${CQLIB}
         fi
         export LIBPATH

         if test "X${LD_LIBRARY_PATH}" != "X" ; then
              LD_LIBRARY_PATH=${CQLIB}:${LD_LIBRARY_PATH}
         else
              LD_LIBRARY_PATH=${CQLIB}
         fi
         export LD_LIBRARY_PATH
         ;;
 
   # solaris
   sunv5_intel)
         if test "X${LD_LIBRARY_PATH}" != "X" ; then
              LD_LIBRARY_PATH=${CQLIB}:${GLLIB}:${GCCLIB}:${LD_LIBRARY_PATH}
         else
              LD_LIBRARY_PATH=${CQLIB}:${GLLIB}:${GCCLIB}
         fi
         export LD_LIBRARY_PATH
         ;;

   sunv5)
         if test "X${LD_LIBRARY_PATH}" != "X" ; then
              LD_LIBRARY_PATH=${CQLIB}:${GLLIB}:${CCLIB}:${LD_LIBRARY_PATH}
         else
              LD_LIBRARY_PATH=${CQLIB}:${GLLIB}:${CCLIB}
         fi
         export LD_LIBRARY_PATH
         ;;

   # set LD_LIBRARY_PATH for everything else

   *)    if test "X${LD_LIBRARY_PATH}" != "X" ; then
              LD_LIBRARY_PATH=${CQLIB}:${GLLIB}:${LD_LIBRARY_PATH}
         else
              LD_LIBRARY_PATH=${CQLIB}:${GLLIB}
         fi
         export LD_LIBRARY_PATH
         ;;

esac 


#Finally, execute the program
${CQROOT}/c_${mach}/bin/${APPLICATION_BASE}.x ${cmdopt}
