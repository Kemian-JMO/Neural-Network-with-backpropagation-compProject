package com.ann.project;

import javax.imageio.*;
import java.awt.image.*;
import java.io.*;
import java.awt.*;

public class Imager {
    private String Zero = "src/main/resources/Pictures/Zero.png";
    private String One = "src/main/resources/Pictures/One.png";
    private String Two = "src/main/resources/Pictures/Two.png";
    private String Three = "src/main/resources/Pictures/Three.png";
    private String Four = "src/main/resources/Pictures/Four.png";
    private String Five = "src/main/resources/Pictures/Five.png";
    private String Six = "src/main/resources/Pictures/Six.png";
    private String Seven = "src/main/resources/Pictures/Seven.png";
    private String Eight = "src/main/resources/Pictures/Eight.png";
    private String Nine = "src/main/resources/Pictures/Nine.png";
    private String[] images = {Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine};
    private int width = 28;
    private int height = 28;




    /*
        make monochrome image
        resize image to 28x28
        make the pixels into a 2d array
        normalize the pixels to be between 0 and 1

     */

    public double[][] getImageData(){
        double[][] imageArray = new double[images.length][width * height];
        for (int i = 0; i < images.length; i++) {
            BufferedImage img = addImage(images[i]);
            double[] nia = getNormalisedImageArray(img);
            imageArray[i] = nia.clone();
        }
        return imageArray;
    }

    public double[][] getImageLabel(){
        double[][] labelArray = new double[images.length][10];
        for (int i = 0; i < images.length; i++) {
            labelArray[i][i] = 1;
        }
        return labelArray;
    }



    /*
    https://www.geeksforgeeks.org/java/image-processing-in-java-read-and-write/
     */
    public double[] getNormalisedImageArray(Image img){
        double[] imageArray = new double[width * height];

        BufferedImage gImg = new BufferedImage(width,height,BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gImg.createGraphics();
        g.drawImage(img, 0, 0, 28,28, null);
        g.dispose();
        for (int i = 0; i < gImg.getWidth(); i++) {
            for (int j = 0; j < gImg.getHeight(); j++) {
                int pixel = gImg.getRGB(i, j) & 0xff;
                double colour = normalize0t1(pixel);
                imageArray[i * gImg.getHeight() + j] = colour;
            }
        }

        return imageArray;
    }

    public double normalize0t1(double pixel){
        double colour = 1;
        return colour - pixel / 255;
    }

    public BufferedImage addImage(String path){
        BufferedImage img = null;
        try {
            File input = new File(path);

            img = ImageIO.read(input);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return img;
    }
}
