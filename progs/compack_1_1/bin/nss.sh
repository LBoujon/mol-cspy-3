#!/bin/sh
# $Id: nss.sh,v 1.4 2006/03/01 15:15:22 darling Exp $

# New and shiney. Still called nss.sh, even though it looks nothing like
# the old one.

# generally invoked by apps as: "nss.sh -remote 'openURL(%1)'"

# List of browsers to check for. To override this, set CCDC_BROWSER to
# the name or full path of your favoured broser in the environment that
# the application is started from
browser_list="firefox mozilla opera konqueror netscape"
last_browser=`echo $browser_list | sed 's;.* \(.*\)$;\1;'`

# don't let ours interfere with the browser
unset LD_LIBRARY_PATH

# set -x
yes="yes"

browser_ok () {
    res=`type "$browser" 2>/dev/null 1>&2; echo $?`
    [ "$res" -eq 0 ] && echo $yes || echo "no"
}

already_running () {
    case "`uname`$1" in
        # old netscapes under IRIX don't respond to pings correctly.
        IRIX*netscape ) { [ -h $HOME/.netscape/lock ] && echo $yes || echo "no"; return; } ;;
    esac
    res1=`$1 -remote 'ping()' 2>&1 | grep window`
    res2=`$1 -remote 'ping()' 2>&1 | grep display`
    [ -z "${res1}${res2}" ] && echo $yes || echo "no"
}

case $# in
    "1" ) # maybe one day someone will call us with just a url to open?
        url="$1"
    ;;
    "2" )
        url=`echo $2 | sed 's;openURL(\(.*\));\1;'`
    ;;
    * )
        echo "$0: unexpected arguments:"
        echo " \"$@\""
        exit 1
    ;;
esac

for browser in "$CCDC_BROWSER" "$RELIBASE_BROWSER"; do
    if [ ! -z "$browser" ]; then
        if [ "`browser_ok "$browser"`" = "$yes" ]; then
            break
        else
            echo "$0:"
            echo "RELIBASE_BROWSER or CCDC_BROWSER is set to a nonexistent browser or is not in \$PATH:"
            echo "   `env | grep "$browser"`"
            browser=""
        fi
    fi
done

if [ -z "$browser" ]; then
    for browser in $browser_list; do
        if [ "`browser_ok "$browser"`" = "$yes" ]; then
            # got a browser, have a kitkat
            echo "$0:"
            echo "invoking `which "$browser"`".
            echo "Set the environment variable CCDC_BROWSER if you don't like this one."
            break;
        fi
        if [ "$browser" = "$last_browser" ]; then
            # got to the end of the list and didn't get anything usable
            echo "$0: couldn't find a browser to run." 1>&2
            echo "Please set the environment variable CCDC_BROWSER to your prefered web browser" 1>&2
            exit 1
        fi
    done
fi

case "$browser" in
    *netscape* | *mozilla* | *firefox* )
        if [ "`already_running "$browser"`" = "$yes" ]; then
            # we can use the '-remote' if it was passed
            url="$@"
        fi
    ;;
    *opera* )
        url="$@"  # works even it's not already running
    ;;
    * )
    ;;
esac

# echo "$browser $url"
exec $browser $url &
