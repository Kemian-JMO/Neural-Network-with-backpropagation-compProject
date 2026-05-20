package com.ann.project;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;

import java.io.IOException;

/*
references code from https://docs.oracle.com/javase/8/javafx/user-interface-tutorial/line-chart.htm#CIHGBCFI
https://www.pragmaticcoding.ca/javafx/MVC_In_JavaFX
 */
public class Launcher extends Application {
    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        Controller controller = new Controller();
        stage.setTitle("Neural Network Training");
        stage.setScene(new Scene(controller.getView(), 1000, 600));
        stage.show();
    }
}

class View extends VBox {
    //defining the axes
    final NumberAxis xAxis = new NumberAxis();
    final NumberAxis yAxis = new NumberAxis();
            xAxis.setLabel("Number of Month");
    //creating the chart
    final LineChart<Number,Number> lineChart =
            new LineChart<Number,Number>(xAxis,yAxis);

            lineChart.setTitle("Stock Monitoring, 2010");
    //defining a series
    XYChart.Series series = new XYChart.Series();
            series.setName("My portfolio");
    //populating the series with data
            series.getData().add(new XYChart.Data(1, 23));
            series.getData().add(new XYChart.Data(2, 14));
            series.getData().add(new XYChart.Data(3, 15));
            series.getData().add(new XYChart.Data(4, 24));
            series.getData().add(new XYChart.Data(5, 34));
            series.getData().add(new XYChart.Data(6, 36));
            series.getData().add(new XYChart.Data(7, 22));
            series.getData().add(new XYChart.Data(8, 45));
            series.getData().add(new XYChart.Data(9, 43));
            series.getData().add(new XYChart.Data(10, 17));
            series.getData().add(new XYChart.Data(11, 29));
            series.getData().add(new XYChart.Data(12, 25));

    Scene scene  = new Scene(lineChart,800,600);
            lineChart.getData().add(series);


    public View(Model model) {
    }
}

class Controller {
    private final Region view;

    Controller() {
        Model model = new Model();
        view = new View(model);
    }

    public Region getView() {
        return view;
    }

    void addFive() {
        try {
            viewModel.setNumber(Integer.toString(Integer.parseInt(viewModel.getNumber()) + 5));
        } catch (NumberFormatException e) {
            viewModel.setNumber("5");
        }
    }

    private boolean checkIfMoreAllowed() {
        try {
            int numberValue = Integer.parseInt(viewModel.getNumber());
            return (numberValue < 21);
        } catch (Exception e) {
            return true;
        }
    }
}

class Model {

    public void networking() throws Exception{
        double[][] images = MnistLoader.loadImages(MnistLoader.TRAIN_IMAGE_FILE);
        double[][] labels = MnistLoader.loadLabels(MnistLoader.TRAIN_LABEL_FILE);
        double[][] valImages = MnistLoader.loadImages(MnistLoader.TEST_IMAGE_FILE);
        double[][] valLabels = MnistLoader.loadLabels(MnistLoader.TEST_LABEL_FILE);
        double[][][] data = {images, labels, valImages, valLabels};
        Activation RelU = new ReLU();
        Activation Sigmoid = new Sigmoid();
        Activation SOFTMAX = new SoftMax();
        Activation[] activations = {null, RelU, RelU, RelU, SOFTMAX};
        int[] hiddenLayers = {200,200,200};

        NeuralNetwork neuralNetwork = new NeuralNetwork(10,10,30,0.01, activations, hiddenLayers, data);
        neuralNetwork.trainEpoch();
        Imager img = new Imager();
        double[][] testImages = img.getImageData();
        double[][] testLabels = img.getImageLabel();
        NeuralNetwork bestNetwork = new NeuralNetwork(NeuralNetwork.loadNetwork("Models/network.nn"));
        bestNetwork.runInferenceSingle(testImages, testLabels);
        }
}



/*
public class LineChartSample extends Application {

    @Override public void start(Stage stage) {
        stage.setTitle("Line Chart Sample");
        //defining the axes
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Number of Month");
        //creating the chart
        final LineChart<Number,Number> lineChart =
                new LineChart<Number,Number>(xAxis,yAxis);

        lineChart.setTitle("Stock Monitoring, 2010");
        //defining a series
        XYChart.Series series = new XYChart.Series();
        series.setName("My portfolio");
        //populating the series with data
        series.getData().add(new XYChart.Data(1, 23));
        series.getData().add(new XYChart.Data(2, 14));
        series.getData().add(new XYChart.Data(3, 15));
        series.getData().add(new XYChart.Data(4, 24));
        series.getData().add(new XYChart.Data(5, 34));
        series.getData().add(new XYChart.Data(6, 36));
        series.getData().add(new XYChart.Data(7, 22));
        series.getData().add(new XYChart.Data(8, 45));
        series.getData().add(new XYChart.Data(9, 43));
        series.getData().add(new XYChart.Data(10, 17));
        series.getData().add(new XYChart.Data(11, 29));
        series.getData().add(new XYChart.Data(12, 25));

        Scene scene  = new Scene(lineChart,800,600);
        lineChart.getData().add(series);

        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
*/