# Diabetes Risk Digital Twin: User Guide

This dashboard is a research prototype. Its predictions and simulations are not
medical diagnoses, treatment recommendations, or proof of cause and effect.

## Start the system

1. Open Docker Desktop and wait until Docker reports that its engine is running.
2. Confirm the Ollama application is running and `qwen2.5-coder:1.5b` is installed.
3. Open PowerShell in this project folder.
4. Run:

   ```powershell
   docker compose up -d dashboard
   ```

5. Open `http://127.0.0.1:5000` in a browser.
6. Confirm that the page displays the active dataset name and patient count.
   Neo4j Browser is available separately at `http://127.0.0.1:7474`.

## Use the dashboard

1. **Select a patient.** Enter a patient number and choose **Load patient**, or
   choose **Use** beside a patient in the numbered table.
2. **Wait for processing.** On the first prediction, model loading and 3D export
   can take one or two minutes.
3. **Read Stage 1.** Review the predicted category and class probabilities.
4. **Read Stage 2.** Review the strongest SHAP factors. These describe the
   model's decision for the selected patient; they are not medical causes.
5. **Inspect Stage 3.** The twin uses the selected patient's BMI and sex, while
   the predicted high-risk probability controls the risk colour. Drag to rotate
   the model and scroll to zoom.
6. **Explore Stage 4.** Use the filters or keyboard node selector, then choose a
   node to focus its immediate connections and open its details. The graph links
   all 21 observations and decoded states to their SHAP contributions, the three
   probabilities, model evaluation, and the current Digital Twin. Patient data
   is not stored in Neo4j, and the graph makes no causal or clinical claims.
7. **Generate optional local guidance.** Choose **Generate local research
   guidance** to send the temporary Stage 4 evidence to Ollama on this computer.
   Treat the text as a fallible model explanation, not medical advice.
8. **Run a what-if comparison.** Change the allowed scenario fields and choose
   **Compare scenario**. The system compares the edited values with the original
   dataset record and displays the current and scenario 3D twins side by side.
9. **Choose another patient.** The prediction, explanation, knowledge pathway,
   and 3D twin update for the new numbered record.

## Import another patient dataset

1. Prepare a CSV containing all 21 BRFSS input columns used by this project.
2. The `Diabetes_012` target column is optional.
3. On the dashboard, choose the CSV under **Import another patient CSV**.
4. Choose **Import dataset**.
5. Check the confirmation message and updated patient count.

The imported dataset is kept in memory. Restarting the dashboard restores the
default project dataset.

## Stop or restart the system

```powershell
docker compose restart dashboard
docker compose stop dashboard neo4j
```

If the page does not load, check:

```powershell
docker compose ps
docker compose logs --tail 100 dashboard
```
