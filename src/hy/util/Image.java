package hy.util;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

import lombok.SneakyThrows;
import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class Image {

    public final String[] ReadableFormats = ImageIO.getReaderFileSuffixes();
    public final String[] WritableFormats = ImageIO.getWriterFileSuffixes();

    public String[] paths(String folder) {
        return Stream.of(new File(folder).listFiles()).map(File::getAbsolutePath)//
                .filter(path -> Stream.of(ReadableFormats).anyMatch(fmt -> path.endsWith(fmt)))//
                .toArray(String[]::new);
    }

    @SneakyThrows
    public NArray read(String path, int width, int height) {
        val bimg = resize(ImageIO.read(new File(path)), width, height);
        val aimg = new NArray(3, width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                val rgb = new Color(bimg.getRGB(i, j));
                aimg[at(0, i, j)] = rgb.getRed() / 255.0;
                aimg[at(1, i, j)] = rgb.getBlue() / 255.0;
                aimg[at(2, i, j)] = rgb.getGreen() / 255.0;
            }
        }
        return aimg;
    }

    @SneakyThrows
    public void write(NArray aimg, int width, int height, String path) {
        val max = aimg.reduce(1.0, (a, b) -> Math.max(Math.abs(a), Math.abs(b)));
        val file = new File(path);
        file.getParentFile().mkdirs();
        aimg = aimg.reshaped(3, width, height);
        val bimg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                val r = clip(aimg[at(0, i, j)] * 255.0 / max);
                val b = clip(aimg[at(1, i, j)] * 255.0 / max);
                val g = clip(aimg[at(2, i, j)] * 255.0 / max);
                val rgb = new Color(r, g, b).getRGB();
                bimg.setRGB(i, j, rgb);
            }
        }
        ImageIO.write(bimg, "png", file);
    }

    public BufferedImage resize(BufferedImage original, int width, int height) {
        BufferedImage scaled = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = scaled.createGraphics();
        g2.drawImage(original, 0, 0, width, height, null);
        g2.dispose();
        return scaled;
    }

    private int clip(double v) {
        return (int) (v < 0.0 ? v + 255.0 : v);
    }

    private int[] at(int... arr) { return arr; }
}
