import React from "react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
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

const OtherDetail = () => {

    useEffect(() => {

    }, [])

    return (
        <div className="home">
            <div>
                <div style={{ paddingTop: 20 }}>
                    <h2 style={{ textAlign: 'center' }}></h2>
                    <Bar
                        data={{
                            labels: ['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score'],
                            datasets: [
                                {
                                    label: 'Score',
                                    data: [0.8699186991869918,0.7,0.75,0.7241379310344827],
                                    backgroundColor: [
                                        'rgba(75, 192, 192, 0.2)',
                                        'rgba(75, 192, 192, 0.2)',
                                        'rgba(75, 192, 192, 0.2)',
                                        'rgba(75, 192, 192, 0.2)'
                                    ],
                                    borderColor: [
                                        'rgba(75, 192, 192, 1)',
                                        'rgba(75, 192, 192, 1)',
                                        'rgba(75, 192, 192, 1)',
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
            <div><img src={require('./images/resim11.jpeg')} style={{ width: 600, height: 600 }} /></div>
        </div>
    );

}
export default OtherDetail