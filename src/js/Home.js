import React, { useState, useEffect } from 'react';

const Home = () => {
    const [data, setData] = useState({});
    const [result, setResult] = useState({ "value": "" });
    const [hitNumber, setHitNumber] = useState({});
    const [predictionResult, setPredictionResult] = useState("");
    const [allSensorsValues, setAllSensorsValues] = useState([]);

    const [realResult, setRealResult] = useState("");
    const [clock, setClock] = useState("");
    const [nameHome, setNameHome] = useState("");
    const [nameAway, setNameAway] = useState("");
    const [sexHome, setSexHome] = useState("Male");
    const [sexAway, setSexAway] = useState("Male");
    const [weightHome,setWeightHome] = useState("");
    const [weightAway,setWeightAway] = useState("");
    const [heightHome, setHeightHome] = useState("");
    const [heightAway, setHeightAway] = useState("");
    const [bmiHome, setBmiHome] = useState("");
    const [bmiAway, setBmiAway] = useState("");
    const [handHome, setHandHome] = useState("Left Hand");
    const [handAway, setHandAway] = useState("Left Hand");
    const [ageHome, setAgeHome] = useState("");
    const [ageAway, setAgeAway] = useState("");
    const [experienceHome, setExperienceHome] = useState("");
    const [experienceAway, setExperienceAway] = useState("");
    const [colName, setColName] = useState("Sex_Home");

    function compareArrays(a, b) {
        for (let i = 0; i < a.length; i++) {
            if (a[i] != b[i]) {
                console.log(a[i] + " " + b[i]);
                return false;
            }
        }
        return true;
    }

    async function getData(abortCont) {
        try {
            console.log("DATA READING STARTED");
            fetch("http://localhost:5000/members?sex_home=" + sexHome + "&sex_away=" + sexAway + "&weight_home=" + weightHome + "&weight_away=" + weightAway + "&height_home=" + heightHome + "&height_away=" + heightAway + "&clock=" + clock + "&hand_home=" + handHome + "&hand_away=" + handAway + "&age_home=" + ageHome + "&age_away=" +ageAway + "&experience_home=" + experienceHome + "&experience_away=" + experienceAway, { signal: abortCont.signal }).then(
                res => res.json()
            ).then(
                d => {
                    var tempData = localStorage.getItem("tempData");
                    tempData ? tempData = JSON.parse(localStorage.getItem("tempData")) : tempData = undefined;
                    console.log(tempData);
                    if (tempData == undefined) {
                        console.log("undefined");
                        localStorage.setItem("tempData", JSON.stringify(d));
                        setData(d);
                        let tempAllSensorsValues = allSensorsValues;
                        tempAllSensorsValues.push(d.sensorValues);
                        setAllSensorsValues(tempAllSensorsValues);
                        let tempHitNumber = JSON.parse(localStorage.getItem("tempHitNumber"));
                        let max = Math.max(...d.sensorValues);
                        let maxIndex = d.sensorValues.indexOf(max);
                        tempHitNumber.values[maxIndex]++; 
                        setHitNumber(tempHitNumber);
                        localStorage.setItem("tempHitNumber", JSON.stringify(tempHitNumber));
                        if (d.prediction == true) {
                            let temp = JSON.parse(localStorage.getItem("tempResult"));
                            temp.value += "1";
                            setResult(temp);
                            setPredictionResult(temp.value);

                        } else {
                            let temp = JSON.parse(localStorage.getItem("tempResult"));
                            temp.value += "0";
                            setResult(temp);
                            setPredictionResult(temp.value);
                        }
                    }
                    else if (!compareArrays(d.sensorValues, tempData.sensorValues) && (d.sensorValues[0] > 0 || d.sensorValues[1]>0 || d.sensorValues[2] || d.sensorValues[3]>0)) {
                        console.log("defined");
                        localStorage.setItem("tempData", JSON.stringify(d));
                        setData(d);
                        let tempAllSensorsValues = allSensorsValues;
                        tempAllSensorsValues.push(d.sensorValues);
                        setAllSensorsValues(tempAllSensorsValues);
                        let tempHitNumber = JSON.parse(localStorage.getItem("tempHitNumber"));
                        let max = Math.max(...d.sensorValues);
                        let maxIndex = d.sensorValues.indexOf(max);
                        tempHitNumber.values[maxIndex]++; 
                        setHitNumber(tempHitNumber);
                        localStorage.setItem("tempHitNumber", JSON.stringify(tempHitNumber));
                        if (d.prediction == true) {
                            let temp = JSON.parse(localStorage.getItem("tempResult"));
                            temp.value += "1";
                            localStorage.setItem("tempResult", JSON.stringify(temp));
                            setResult(temp);
                            setPredictionResult(temp.value);
                        } else {
                            let temp = JSON.parse(localStorage.getItem("tempResult"));
                            temp.value += "0";
                            localStorage.setItem("tempResult", JSON.stringify(temp));
                            setResult(temp);
                            setPredictionResult(temp.value);
                        }
                    } else {
                        console.log("NO CHANGING");
                    }
                }
            );
        } catch (ex) { console.log(ex); }

    }

    useEffect(() => {
        localStorage.setItem("tempResult", JSON.stringify({ "value": "" }));
        localStorage.setItem("tempHitNumber", JSON.stringify({ "values": [0, 0, 0, 0] }));
    }, []);

    const startGame = (e) => {
        e.preventDefault();
        const abortCont = new AbortController();

        setInterval(() => {
            getData(abortCont);
        }, 200);
        return () => abortCont.abort();
    }

    const handleSubmit = (e) => {
        e.preventDefault();
        let regionHitCount = [hitNumber.values[0], hitNumber.values[1], hitNumber.values[2], hitNumber.values[3]];
        let dateTime = new Date().toLocaleString();
        let abTestResultTemp;

        fetch("http://localhost:5000/getabtest?col=" + colName).then(
            res => res.json()
        ).then(
            d => {
                abTestResultTemp = d;
                return abTestResultTemp;
            }
        ).then(a => {
            let abTestResult = a.abTestResult;
            let abTestColName = a.abTestColName;
            const myStatistic = { dateTime, allSensorsValues, regionHitCount, predictionResult, realResult, clock, nameHome, nameAway, sexHome, sexAway, bmiHome, bmiAway, handHome, handAway, ageHome, ageAway, experienceHome, experienceAway, abTestResult,abTestColName };
            fetch('http://localhost:8000/myStatistics', {
                method: 'POST',
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(myStatistic)
            }).then(() => {
                console.log("-------------------New Statistic Added---------------------");
            })

            console.log(myStatistic);
        });


    };

    return (
        <div className="home">
            <h2>Live Data Stream</h2>
            <div style={{ paddingBottom: 20 }}>
                <table border="1" style={{ width: 600 }}>
                    <thead>
                        <tr>
                            <th>Region 1</th>
                            <th>Region 2</th>
                            <th>Region 3</th>
                            <th>Region 4</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>{
                            (typeof data.sensorValues === undefined) ? (
                                <p>loading</p>
                            ) : (
                                data.sensorValues?.map((value, i) => (
                                    <td key={i}>{value}</td>
                                ))
                            )
                        }
                        </tr>

                    </tbody>
                </table>
            </div>
            <div style={{ paddingBottom: 20 }}>
                <table border="1" style={{ width: 600 }}>
                    <thead>
                        <tr>
                            <th># of hits on region 1</th>
                            <th># of hits on region 2</th>
                            <th># of hits on region 3</th>
                            <th># of hits on region 4</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>{
                            (typeof hitNumber.values === undefined) ? (
                                <p>loading</p>
                            ) : (
                                hitNumber.values?.map((value, i) => (
                                    <td key={i}>{value}</td>
                                ))
                            )
                        }
                        </tr>
                    </tbody>
                </table>
            </div>


            <div style={{ paddingTop: 5 }}>{
                (typeof data.sensorValues === undefined) ? (
                    <p>loading</p>
                ) : (
                    data.prediction ? <p style={{ fontWeight: "bolder" }}>Prediction : Successful</p> : <p style={{ fontWeight: "bolder" }}> Prediction : Unsuccessful</p>
                )
            }</div>

            <label>Prediction Result : {result.value}</label>
            <br />
            <button onClick={startGame}>start</button>
            <div className="create" style={{ paddingTop: 10 }}>
                <form onSubmit={handleSubmit}>
                    <label>Real Result:</label>
                    <input
                        type="text"
                        required
                        value={realResult}
                        onChange={(e) => setRealResult(e.target.value)} />

                    <label>*Clock:</label>
                    <input
                        type="text"
                        required
                        value={clock}
                        onChange={(e) => setClock(e.target.value)} />

                    <label>Name-Home:</label>
                    <input
                        type="text"
                        required
                        value={nameHome}
                        onChange={(e) => setNameHome(e.target.value)} />
                    <label>Name-Away:</label>
                    <input
                        type="text"
                        required
                        value={nameAway}
                        onChange={(e) => setNameAway(e.target.value)} />

                    <label>*Sex-Home:</label>
                    <select
                        value={sexHome}
                        onChange={(e) => setSexHome(e.target.value)}>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>

                    <label>*Sex-Away:</label>
                    <select
                        value={sexAway}
                        onChange={(e) => setSexAway(e.target.value)}>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>

                    <label>*Weight-Home:</label>
                    <input
                        type="text"
                        required
                        value={weightHome}
                        onChange={(e) => setWeightHome(e.target.value)} />

                    <label>*Weight-Away:</label>
                    <input
                        type="text"
                        required
                        value={weightAway}
                        onChange={(e) => setWeightAway(e.target.value)} />

                    <label>*Height-Home:</label>
                    <input
                        type="text"
                        required
                        value={heightHome}
                        onChange={(e) => setHeightHome(e.target.value)} />

                    <label>*Height-Away:</label>
                    <input
                        type="text"
                        required
                        value={heightAway}
                        onChange={(e) => setHeightAway(e.target.value)} />

                    <label>BMI-Home:</label>
                    <input
                        type="text"
                        required
                        value={bmiHome}
                        onChange={(e) => setBmiHome(e.target.value)} />
                    <label>BMI-Away:</label>
                    <input
                        type="text"
                        required
                        value={bmiAway}
                        onChange={(e) => setBmiAway(e.target.value)} />

                    <label>*Hand-Home:</label>
                    <select
                        value={handHome}
                        onChange={(e) => setHandHome(e.target.value)}>
                        <option value="Left Hand">Left Hand</option>
                        <option value="Right Hand">Right Hand</option>
                    </select>
                    <label>*Hand-Away:</label>
                    <select
                        value={handAway}
                        onChange={(e) => setHandAway(e.target.value)}>
                        <option value="Left Hand">Left Hand</option>
                        <option value="Right Hand">Right Hand</option>
                    </select>

                    <label>*Age-Home:</label>
                    <input
                        type="text"
                        required
                        value={ageHome}
                        onChange={(e) => setAgeHome(e.target.value)} />
                    <label>*Age-Away:</label>
                    <input
                        type="text"
                        required
                        value={ageAway}
                        onChange={(e) => setAgeAway(e.target.value)} />

                    <label>*Experience-Home:</label>
                    <input
                        type="text"
                        required
                        value={experienceHome}
                        onChange={(e) => setExperienceHome(e.target.value)} />
                    <label>*Experience-Away:</label>
                    <input
                        type="text"
                        required
                        value={experienceAway}
                        onChange={(e) => setExperienceAway(e.target.value)} />

                    <label>A/B Test Col:</label>
                    <select
                        value={colName}
                        onChange={(e) => setColName(e.target.value)}>
                        <option value="Sex_Home">Sex_Home</option>
                        <option value="Sex_Away">Sex_Away</option>
                        <option value="Left_hand_Home">Left_hand_Home</option>
                        <option value="Left_hand_Away">Left_hand_Away</option>
                    </select>

                    <button>Save</button>
                </form>
            </div>


        </div>
    );
}
export default Home;