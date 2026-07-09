// Human Development Index Prediction Client Logic

document.addEventListener("DOMContentLoaded", () => {
    const hdiForm = document.getElementById("hdiForm");
    
    if (hdiForm) {
        hdiForm.addEventListener("submit", (event) => {
            const country = document.getElementById("country").value;
            const lifeExp = parseFloat(document.getElementById("life_exp").value);
            const meanSchool = parseFloat(document.getElementById("mean_school").value);
            const gni = parseFloat(document.getElementById("gni").value);
            const internet = parseFloat(document.getElementById("internet").value);
            
            // Basic client-side validation
            if (!country) {
                alert("Please select a country.");
                event.preventDefault();
                return;
            }
            
            if (isNaN(lifeExp) || lifeExp < 50 || lifeExp > 85) {
                alert("Life Expectancy must be between 50 and 85.");
                event.preventDefault();
                return;
            }
            
            if (isNaN(meanSchool) || meanSchool < 1 || meanSchool > 15) {
                alert("Mean Years of Schooling must be between 1 and 15.");
                event.preventDefault();
                return;
            }
            
            if (isNaN(gni) || gni < 290 || gni > 140000) {
                alert("Gross National Income must be between $290 and $140,000.");
                event.preventDefault();
                return;
            }
            
            if (isNaN(internet) || internet < 1 || internet > 100) {
                alert("Internet Users percentage must be between 1% and 100%.");
                event.preventDefault();
                return;
            }
            
            // Add a loading class to the submit button
            const submitBtn = hdiForm.querySelector("button[type='submit']");
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing Prediction...';
            }
        });
    }
});
