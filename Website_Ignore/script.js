// JS 1: 3D Animation (TiltEffect)
    const card = document.querySelector('.main-card');
    const container = document.querySelector('.container');

    // container.addEventListener('mousemove', (e) => {
    //     let xAxis = (window.innerWidth / 2 - e.pageX) / 40; // Κίνηση στον X άξονα
    //     let yAxis = (window.innerHeight / 2 - e.pageY) / 40; // Κίνηση στον Y άξονα
    //     card.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
    // });

    // Reset το animation στο mouseout
    container.addEventListener('mouseleave', (e) => {
        card.style.transform = `rotateY(0deg) rotateX(0deg)`;
    });


    document.getElementById('predictBtn').addEventListener('click', async () => {
        const homeTeam = document.getElementById('home_team').value;
        const awayTeam = document.getElementById('away_team').value;
        const loading = document.getElementById('loading');
        const resultSection = document.getElementById('result-section');

        // Έλεγχος αν επιλέχθηκε η ίδια ομάδα
        if (homeTeam === awayTeam) {
            alert("Please choose different teams.");
            return;
        }

        // Εμφάνιση loading
        loading.style.display = 'block';
        resultSection.classList.remove('show');


        const requestData = {
            home_team: homeTeam,
            away_team: awayTeam,
            home_days_rest: 7, // Default τιμή για ευκολία
            away_days_rest: 7
        };

    

        try {

            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });

            if (!response.ok) throw new Error('Error while connecting to the API');

            const result = await response.json();


            document.getElementById('finalPrediction').innerText = result.Prediction;
            document.getElementById('probHome').innerText = result.Home_Win_Prob;
            document.getElementById('probDraw').innerText = result.Draw_Prob;
            document.getElementById('probAway').innerText = result.Away_Win_Prob;


            loading.style.display = 'none';
            resultSection.classList.add('show');

        } catch (error) {
            console.error(error);
            alert("Error while trying to connect with model .");
            loading.style.display = 'none';
        }
    });