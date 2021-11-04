package com.example.refresh_selection;

import android.view.View;

public class ScheduleCard {//
    private String schedule_date;
    private View schedule;

//    public ScheduleCard(String schedule_date,View schedule){
//        this.schedule_date=schedule_date;
//        this.schedule=schedule;
//    }
    public ScheduleCard(String schedule_date){
        this.schedule_date=schedule_date;
    }

    public String getSchedule_date() {
        return this.schedule_date;
    }

    public void setSchedule_date(String schedule_date) {
        this.schedule_date = schedule_date;
    }

    public View getSchedule() {
        return this.schedule;
    }

    public void setSchedule(View schedule) {
        this.schedule = schedule;
    }
}
