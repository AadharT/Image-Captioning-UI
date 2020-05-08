using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Windows.Forms;
using System.Diagnostics;


namespace AutoCaption
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog opnfd = new OpenFileDialog();
            opnfd.Filter = "Image Files (*.jpg;*.jpeg;.*.gif;)|*.jpg;*.jpeg;.*.gif";
            if (opnfd.ShowDialog() == DialogResult.OK)
            {
                imageName.Text = opnfd.FileName;
                pictureBox1.Image = new Bitmap(opnfd.FileName);
            }
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
        private void get_caption(string ImagePath)
        {
            // Put your python script path here, don't remove the @
            string fileName = @"final.py";
            fileName += ' ';
            fileName += ImagePath;
            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(@"python", fileName)
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            p.Start();

            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit();

            Console.WriteLine(output);
            result.Text = output;
            Console.ReadLine();
        }
        private void button2_Click(object sender, EventArgs e)
        {
            string imagePath = imageName.Text;
            get_caption(imagePath);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}
