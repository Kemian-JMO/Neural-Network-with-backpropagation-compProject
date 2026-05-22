package com.ann.project;
import javafx.application.Application;
import javafx.beans.binding.Bindings;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;

import javax.swing.text.View;
import java.io.IOException;

/*
references code from https://docs.oracle.com/javase/8/javafx/user-interface-tutorial/line-chart.htm#CIHGBCFI
https://www.pragmaticcoding.ca/javafx/MVC_In_JavaFX
 */

/*
public class Launcher extends Application {
    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        Coordinator coordinator = new Coordinator();
        stage.setTitle("Neural Network Training");
        stage.setScene(new Scene(controller.getView(), 1000, 600));
        stage.show();
    }
}

class Coordinator{
    private final Model model;
    private final ChartView chartView;
    private final SettingsView settingsView;
    private final Controller controller;

}

class ChartView extends BorderPane {

}

class SettingsView extends BorderPane {
    private Spinner<Integer> epochsSpinner;
    private Spinner<Integer> batchSizeSpinner;
    private Slider learningRateSlider;
    private Label learningRateLabel;
    private Spinner<Integer> hiddenLayerSpinner;
    private HBox activationRow;
    private Button startButton;
    private LineChart<Number, Number> lineChart;

    public View(Model model) {
        lineChart = buildChart();
        VBox form = buildForm();
        setCenter(lineChart);
        setLeft(form);
    }

    public Button getStartButton() { return startButton; }
    public Spinner<Integer> getEpochsSpinner() { return epochsSpinner; }
    public Spinner<Integer> getBatchSizeSpinner() { return batchSizeSpinner; }
    public Slider getLearningRateSlider() { return learningRateSlider; }
    public Spinner<Integer> getHiddenLayerSpinner() { return hiddenLayerSpinner; }

    private LineChart<Number, Number> buildChart() {
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Epoch");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Loss / Accuracy");

        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle("Training progress");

        XYChart.Series<Number, Number> fakeSeries = new XYChart.Series<>();
        fakeSeries.setName("Fake training loss");
        fakeSeries.getData().add(new XYChart.Data<>(1, 2.3));
        fakeSeries.getData().add(new XYChart.Data<>(2, 1.8));
        fakeSeries.getData().add(new XYChart.Data<>(3, 1.4));
        fakeSeries.getData().add(new XYChart.Data<>(4, 1.1));
        chart.getData().add(fakeSeries);
        return chart;
    }

    private VBox buildForm() {
        VBox form = new VBox(10);  // 10px spacing
        form.setPadding(new Insets(15));
        form.setPrefWidth(280);

        epochsSpinner = new Spinner<>(1, 1000, 10);
        epochsSpinner.setEditable(true);
        commitOnFocusLoss(epochsSpinner);

        batchSizeSpinner = new Spinner<>(1, 1000, 30);
        batchSizeSpinner.setEditable(true);
        commitOnFocusLoss(batchSizeSpinner);

        learningRateSlider = new Slider(-4.0, 0.0, -2.0);
        learningRateLabel = new Label();
        learningRateLabel.textProperty().bind(
                learningRateSlider.valueProperty().asString("lr = %.5f")
                // Note: this displays the exponent, not the lr. Fix below.
        );
        // Better: derived binding
        learningRateLabel.textProperty().bind(
                Bindings.createStringBinding(
                        () -> String.format("lr = %.5f", Math.pow(10, learningRateSlider.getValue())),
                        learningRateSlider.valueProperty()
                )
        );

        hiddenLayerSpinner = new Spinner<>(1, 10, 3);
        hiddenLayerSpinner.setEditable(true);
        commitOnFocusLoss(hiddenLayerSpinner);

        activationRow = new HBox(5);
        rebuildActivationRow(3);  // initial state

        startButton = new Button("Start Training");

        form.getChildren().addAll(
                new Label("Epochs:"), epochsSpinner,
                new Label("Batch size:"), batchSizeSpinner,
                new Label("Learning rate (log scale):"), learningRateSlider, learningRateLabel,
                new Label("Hidden layers:"), hiddenLayerSpinner,
                new Label("Activations:"), activationRow,
                startButton
        );
        return form;
    }

    public void rebuildActivationRow(int count) {
        activationRow.getChildren().clear();
        for (int i = 0; i < count; i++) {
            ComboBox<ActivationChoice> combo = new ComboBox<>();
            combo.getItems().addAll(ActivationChoice.RELU, ActivationChoice.SIGMOID, ActivationChoice.SOFTMAX);
            combo.setValue(ActivationChoice.RELU);
            activationRow.getChildren().add(combo);
        }
    }

    enum ActivationChoice {
        RELU, SIGMOID, SOFTMAX;

        public Activation toActivation() {
            return switch (this) {
                case RELU -> new ReLU();
                case SIGMOID -> new Sigmoid();
                case SOFTMAX -> new SoftMax();
            };
        }
    }

    private static <T> void commitOnFocusLoss(Spinner<T> spinner) {
        spinner.focusedProperty().addListener((obs, oldVal, newVal) -> {
            if (!newVal) spinner.increment(0);
        });
    }

}

class Controller {
    private final Model model;
    private final View view;

    Controller() {
        this.model = new Model();
        this.view = new View(model);
        view.getStartButton().setOnAction(e -> handleStart());
        view.getHiddenLayerSpinner().valueProperty().addListener(
                (obs, oldVal, newVal) -> view.rebuildActivationRow(newVal)
        );
    }

    public Region getView() { return view; }

    private void handleStart() {
        // slice 1: just read values and print
        int epochs = view.getEpochsSpinner().getValue();
        int batchSize = view.getBatchSizeSpinner().getValue();
        double lr = Math.pow(10, view.getLearningRateSlider().getValue());
        int hiddenCount = view.getHiddenLayerSpinner().getValue();
        System.out.println("Would train: epochs=" + epochs
                + ", batchSize=" + batchSize
                + ", lr=" + lr
                + ", hiddenLayers=" + hiddenCount);
    }
}

class Model {
    private int epochs = 10;
    private int batchSize = 30;
    private double learningRate = 0.01;
    private int hiddenLayerCount = 3;

    public int getEpochs() { return epochs; }
    public void setEpochs(int v) { this.epochs = v; }
}
*/



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