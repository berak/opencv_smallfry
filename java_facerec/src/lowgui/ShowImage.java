package lowgui;

import org.opencv.core.*;
import java.awt.*;
import java.awt.image.*;

/**
*   a mimimal awt Panel to show cv::Mat in java
**/

public class ShowImage extends Panel {
    BufferedImage  image;

    public ShowImage(Mat m) { set(m); }
    public ShowImage() { set(new Mat(10,10,16,new Scalar(200,0,0))); }

    public void set(Mat m) {
        image = bufferedImage(m);
        repaint();
    }

    public void paint(Graphics g) {
        g.drawImage( image, 0, 0, null);
    }

    public static BufferedImage bufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels()*m.cols()*m.rows();
        byte [] pixels = new byte[bufferSize];
        m.get(0,0,pixels); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
        System.arraycopy(pixels, 0, targetPixels, 0, pixels.length);
        return image;
    }
}
