#!/bin/sh

uname=''
if test -f /bin/uname ; then
    uname=/bin/uname
else
    ( uname ) >/dev/null 2>&1
    if test $? -eq 0 ; then
        uname=uname
    fi
fi

longlinux=0

while test $# -ne 0  ; do
    case "$1" in
    -long) longlinux=1 ;;
    *)  echo "Command line option $1 not known"
        exit 1
        ;;
    esac
    shift
done

if test "X$uname" != "X" ; then
  set `$uname -a`

  case "$1" in
    Linux*|linux*) mach_type=linux ;;
    SunOS)
        case "$5" in
            i86pc)         mach_type=sunv5_intel ;;
            *)             mach_type=sunv5 ;;
        esac ;;
    AIX*)                  mach_type=rs6000 ;;
    IRIX*)                 mach_type=sgiv6 ;;
    *)                     mach_type=unsupported ;;
  esac

  if test $longlinux -eq "1" ; then
    if test "x$mach_type" = "xlinux" ; then
      if test -r /etc/redhat-release ; then
        # Check whether this is RedHat Linux 9 or RedHat Enterprise
        case `cut -f3,5 -d' ' /etc/redhat-release` in
          "Linux 9")       mach_type="linux_rh9";;
          "Enterprise WS") mach_type="linux_re3";;
        esac
      fi 
    fi
  fi

  if test "X$mach_type" != "X" ; then
    echo $mach_type
    exit 0
  fi

fi

echo "unknown"
exit 0
