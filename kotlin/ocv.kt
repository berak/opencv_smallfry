import org.opencv.*
import org.opencv.core.*
import org.opencv.videoio.*

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.lang.Thread.*;
import javax.swing.JFrame;
import java.lang.Thread.*;


class ShowImage(): Panel() {
    var image: BufferedImage = bufferedImage(Mat(10,10,16,Scalar(200.0,0.0,0.0)));
    override fun paint(g: Graphics) {
        g.drawImage( this.image, 0, 0, null);
    }
    fun set(m: Mat) {
        this.image = bufferedImage(m);
        repaint();
    }
    fun bufferedImage(m: Mat): BufferedImage  {
        var type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        var image = BufferedImage(m.cols(), m.rows(), type)
        var D = image.getRaster().getDataBuffer() as (DataBufferByte)
        m.get(0, 0, D.getData())
        return image
    }
}

class WaitKey():  KeyListener {
    var k: Int=-1;
    fun get(sleepMillis: Int): Int {
        var millis = sleepMillis
        if (millis<0) millis = 777777777;
        while (millis>0) {
            try { Thread.sleep(50L) }
            catch(x: Exception) {}
            millis -= 50
            if (k >= 0) break
        }
        var k2 = k;
        k = -1;
        return k2;
    }
    override fun keyTyped(e: KeyEvent) {}
    override fun keyPressed(e: KeyEvent) { k=e.getKeyCode() }
    override fun keyReleased(e: KeyEvent) {}
}

class NamedWindow: JFrame
{
    var _imshow = ShowImage()
    var _waitkey = WaitKey()
	constructor(name: String): super(name) {
	    this.setSize(640, 480)
	    setVisible(true)
	    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
	    getContentPane().add(_imshow)
	    addKeyListener(_waitkey)
	}
    fun imshow(m: Mat) {
        _imshow.set(m)
    }
    fun waitKey(t: Int): Int {
        return _waitkey.get(t)
    }
}


fun doc(cls: String, what: Any) {
	var c = Class.forName(cls)
	println(c.getName() + " :")
		when(what) {
		"F" -> { var a = c.getFields();			  for (i in a) println(i.toString()) }
		"M" -> { var a = c.getDeclaredMethods();  for (i in a) println(i.toString()) }
		"C" -> { var a = c.getDeclaredClasses();  for (i in a) println(i.toString()) }
	}
}


fun ocv_init() {
	System.setProperty("java.library.path", "c:/p/kotlinc/bin31/Release/")
	System.setProperty("java.awt.headless", "false")
	//System.loadLibrary(Core.NATIVE_LIBRARY_NAME) // nope ;( !!!!!
	val lp = System.getProperty("java.library.path")
	System.load(lp + Core.NATIVE_LIBRARY_NAME + ".dll")
}


fun test() {
	ocv_init()
    var gui = NamedWindow("kotlin");
	var cap = VideoCapture(0)
    while ( cap.isOpened()) {
    	var m = Mat();
    	if (!cap.read(m))
            break
    	gui.imshow(m);
    	val k = gui.waitKey(50)
    	if (k > -1)
            break
    }
    gui.dispose()
    cap.release()
}
