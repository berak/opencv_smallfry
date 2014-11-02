package lowgui;

import java.awt.event.*;
import java.lang.Thread.*;


/**
* this behaves *slightly* different than highgui's  waitKey:
*   sleepMillis >  0 : wait millis 
*   sleepMillis == 0 : do *not* wait (or check further keys, until you call it again)
*   sleepMillis <  0 : wait forever 
**/

public class WaitKey extends KeyAdapter {
    private int k=-1;
    
    public int get() { return get(-1); }
    
    public int get(int sleepMillis) {
        if (sleepMillis<0) sleepMillis = 777777777;
        while ((sleepMillis>0)&&(k<0)) {
            int t = Math.min(sleepMillis, 50); // give peace a chance.
            try { Thread.sleep(t); } catch(Exception x) {}
            sleepMillis -= t;
        }
        int k2 = k;
        k = -1;
        return k2;
    }   
    
    public void keyPressed(KeyEvent e) { k=e.getKeyChar(); }
}

