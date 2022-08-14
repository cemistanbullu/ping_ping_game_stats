import React from "react";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Bar, Line, Pie } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    LineElement,
    PointElement,
    ArcElement
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    PointElement,
    LineElement,
    ArcElement,
    Title,
    Tooltip,
    Legend
);

const StatisticDetail = () => {
    const { id } = useParams()
    const [myStatistic, setMyStatistic] = useState({});
    const [regionHitCount, setRegionHitCount] = useState([]);
    const [hitCountArray, setHitCountArray] = useState([]);
    const [hitTotalPowerArray, setHitTotalPowerArray] = useState([]);
    const [predictionHomeScore, setPredictionHomeScore] = useState(0);
    const [predictionAwayScore, setPredictionAwayScore] = useState(0);
    const [realHomeScore, setRealHomeScore] = useState(0);
    const [realAwayScore, setRealAwayScore] = useState(0);
    const [predictionColors, setPredictionColors] = useState([]);
    const [realColors, setRealColors] = useState([]);

    useEffect(() => {
        fetch('http://localhost:8000/myStatistics/' + id)
            .then(res => {
                return res.json();
            }).then(data => {
                setMyStatistic(data);
                return data;
            })
            .then((d) => {
                console.log(d)
                let tempPredictionHomeScore = 0;
                let tempPredictionAwayScore = 0;
                let tempRealHomeScore = 0;
                let tempRealAwayScore = 0;
                let tempPredictionColors = [];
                let tempRealColors = [];
                for (let i = 0; i < d.allSensorsValues.length; i++) {
                    let tempHitCountArray = hitCountArray;
                    tempHitCountArray.push(i + 1);
                    setHitCountArray(tempHitCountArray);
                    let hitTotalPower = d.allSensorsValues[i][0] + d.allSensorsValues[i][1] + d.allSensorsValues[i][2] + d.allSensorsValues[i][3];
                    let tempHitTotalPowerArray = hitTotalPowerArray;
                    tempHitTotalPowerArray.push(hitTotalPower);
                    setHitTotalPowerArray(tempHitTotalPowerArray);


                    if (d.predictionResult[i] === "1") {
                        tempPredictionHomeScore++;
                        setPredictionHomeScore(tempPredictionHomeScore);
                        tempPredictionColors.push("green");
                        setPredictionColors(tempPredictionColors);
                    } else if (d.predictionResult[i] === "0") {
                        tempPredictionAwayScore++;
                        setPredictionAwayScore(tempPredictionAwayScore);
                        tempPredictionColors.push("red");
                        setPredictionColors(tempPredictionColors);
                    }
                    if (d.realResult[i] === "1") {
                        tempRealHomeScore++;
                        setRealHomeScore(tempRealHomeScore);
                        tempRealColors.push("green");
                        setRealColors(tempRealColors);
                    } else if (d.realResult[i] === "0") {
                        tempRealAwayScore++;
                        setRealAwayScore(tempRealAwayScore);
                        tempRealColors.push("red");
                        setRealColors(tempRealColors);
                    }

                }
            }

            )

    }, [])

    return (
        <div>
            <div>--------------------------------------------------------------------------------------------</div>
            <div>
                <div style={{ float: 'left' }}>
                    <table style={{ float: 'right', width: 132 }}>
                        <thead>
                            <tr>
                                <th><p style={{ textAlign: 'center' }}></p>Features</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><h4 style={{ textAlign: 'center' }}>Name</h4></td>
                            </tr>
                            <tr>
                                <td><h4 style={{ textAlign: 'center' }}>Age</h4></td>
                            </tr>
                            <tr>
                                <td><h4 style={{ textAlign: 'center' }}>Sex</h4></td>
                            </tr>
                            <tr>
                                <td><h4 style={{ textAlign: 'center' }}>BMI</h4></td>
                            </tr>
                            <tr>
                                <td><h4 style={{ textAlign: 'center' }}>Hand</h4></td>
                            </tr>
                            <tr>
                                <td><h4 style={{ textAlign: 'center' }}>Experience</h4></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div style={{ float: 'right' }}>
                    <table style={{ width: 468 }}>
                        <thead>
                            <tr>
                                <th><p style={{ textAlign: 'center' }}>Home</p></th>
                                <th><p style={{ textAlign: 'center' }}>Away</p></th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.nameHome}</p></td>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.nameAway}</p></td>
                            </tr>
                            <tr>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.ageHome}</p></td>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.ageAway}</p></td>
                            </tr>
                            <tr>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.sexHome}</p></td>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.sexAway}</p></td>
                            </tr>
                            <tr>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.bmiHome}</p></td>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.bmiAway}</p></td>
                            </tr>
                            <tr>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.handHome}</p></td>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.handAway}</p></td>
                            </tr>
                            <tr>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.experienceHome}</p></td>
                                <td><p style={{ textAlign: 'center' }}>{myStatistic.experienceAway}</p></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            <div>--------------------------------------------------------------------------------------------</div>
            <div style={{ width: 599, textAlign: 'center'}}><text style={{ fontWeight: "bolder" }}>     A/B test result of total piezzo values according to "{myStatistic.abTestColName}" : <br></br></text><label>{myStatistic.abTestResult}</label></div>
            <div>--------------------------------------------------------------------------------------------</div>
            <div>
                <div style={{ paddingTop: 20 }}>
                    <h2 style={{ textAlign: 'center' }}># Of Hits On All Regions Of The Racket</h2>
                    <Pie
                        data={{
                            labels: ['First Region', 'Second Region', 'Third Region', 'Forth Region'],
                            datasets: [
                                {
                                    label: '# of hits',
                                    data: myStatistic.regionHitCount,
                                    backgroundColor: [
                                        'rgba(255, 99, 132, 0.2)',
                                        'rgba(54, 162, 235, 0.2)',
                                        'rgba(255, 206, 86, 0.2)',
                                        'rgba(75, 192, 192, 0.2)'
                                    ],
                                    borderColor: [
                                        'rgba(255, 99, 132, 1)',
                                        'rgba(54, 162, 235, 1)',
                                        'rgba(255, 206, 86, 1)',
                                        'rgba(75, 192, 192, 1)'
                                    ],
                                    borderWidth: 1

                                },
                            ],
                        }}
                        height={300}
                        width={590}
                        options={{
                            responsive: false,
                        }}
                    />
                </div>
            </div>
            <div style={{ paddingTop: 30 }}>
                <h2 style={{ textAlign: 'center' }}>Power of Hits With Prediction Score</h2>
                <p style={{ textAlign: 'center' }}>*Green Is Successful, Red Is Unsuccessful Hit</p>
                <p style={{ textAlign: 'center' }}>Power/Hit No</p>
                <Line
                    data={{
                        labels: hitCountArray,
                        datasets: [
                            {
                                label: 'power of hits',
                                data: hitTotalPowerArray,
                                backgroundColor: predictionColors,
                                borderWidth: 1,
                                borderColor: 'grey',

                            },
                        ],
                    }}
                    height={200}
                    width={500}
                    options={{
                        plugins: {
                            legend: {
                                display: false
                            },
                        }
                    }}
                />
            </div>
            <div style={{ paddingTop: 30 }}>
                <h2 style={{ textAlign: 'center' }}>Power of Hits With Real Score</h2>
                <p style={{ textAlign: 'center' }}>*Green Is Successful, Red Is Unsuccessful Hit</p>
                <p style={{ textAlign: 'center' }}>Power/Hit No</p>
                <Line
                    data={{
                        labels: hitCountArray,
                        datasets: [
                            {
                                label: 'Prediction Result',
                                data: hitTotalPowerArray,
                                backgroundColor: realColors,
                                borderWidth: 1,
                                borderColor: 'grey',
                            },
                        ],
                    }}
                    height={200}
                    width={500}
                    options={{
                        plugins: {
                            legend: {
                                display: false
                            },
                        }
                    }}
                />
            </div>
        </div>

    );
}

export default StatisticDetail