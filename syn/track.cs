using System;
using System.Threading;
using System.Drawing;
using System.Windows.Forms;

public class Form1 : System.Windows.Forms.Form
{
    private System.Windows.Forms.TrackBar trackBar1;
    private System.Windows.Forms.TextBox textBox1;
    private System.Windows.Forms.TextBox textBox2;
    private  System.Threading.Thread listen = null;

    [STAThread]
    static void Main()
    {
        Application.Run(new Form1());
    }


    public Form1()
    {
        this.textBox1 = new System.Windows.Forms.TextBox();
        this.textBox2 = new System.Windows.Forms.TextBox();
        this.trackBar1 = new System.Windows.Forms.TrackBar();

        this.textBox1.Location = new System.Drawing.Point(240, 16);
        this.textBox1.Size = new System.Drawing.Size(48, 20);
        this.textBox2.Location = new System.Drawing.Point(240 + 50, 16);
        this.textBox2.Size = new System.Drawing.Size(48, 20);

        this.trackBar1.Location = new System.Drawing.Point(8, 8);
        this.trackBar1.Size = new System.Drawing.Size(224, 45);
        this.trackBar1.Scroll += new System.EventHandler(this.trackBar1_Scroll);
        this.trackBar1.Maximum = 100;
        this.trackBar1.TickFrequency = 5;
        this.trackBar1.LargeChange = 3;
        this.trackBar1.SmallChange = 2;

        // Set up how the form should be displayed and add the controls to the form.
        this.ClientSize = new System.Drawing.Size(360, 62);
        this.Controls.AddRange(new System.Windows.Forms.Control[] {this.textBox1,this.textBox2,this.trackBar1});
        this.Text = "TrackBar Example";

        listen = new Thread(new ThreadStart(delegate()
        {
            while (true)
            {
                string cin = Console.ReadLine();
                Console.WriteLine(cin);
            }
        }));
        listen.Start();
    }

    private void trackBar1_Scroll(object sender, System.EventArgs e)
    {
        Int32 ev = Int32.Parse(textBox1.Text);
        Int32 to = Int32.Parse(textBox2.Text);
        Double frq = (Double)(trackBar1.Value) / 100;
        string str = (to + " " + ev + " " + frq + "\n").Replace(",",".");
        Console.Write(str);
    }
}