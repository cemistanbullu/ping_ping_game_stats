import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const StatisticList = () => {
    const [myStatistics, setMyStatistics] = useState(null);

    useEffect(() => {
        fetch('http://localhost:8000/myStatistics')
            .then(res => {
                return res.json();
            }).then(data => {
                setMyStatistics(data);
            })

    }, [])
    
    return (
        <div className="home">
            {
                myStatistics?.map((myStatistic) => (
                    <div className="blog-preview" key={myStatistic.id}>
                        <Link to={`/statisticdetail/${myStatistic.id}`}>
                            <h2>ID : {myStatistic.id} -- &gt; {myStatistic.nameHome} vs {myStatistic.nameAway} - {myStatistic.dateTime}</h2>
                        </Link>
                    </div>
                ))
            }
        </div>
    );

}
export default StatisticList