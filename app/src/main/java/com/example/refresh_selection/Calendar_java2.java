package com.example.refresh_selection;

import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;

import com.google.android.material.datepicker.MaterialDatePicker;
import com.prolificinteractive.materialcalendarview.CalendarDay;
import com.prolificinteractive.materialcalendarview.MaterialCalendarView;
import com.prolificinteractive.materialcalendarview.OnRangeSelectedListener;

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.util.Pair;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

public class Calendar_java2 extends AppCompatActivity {
    ScheduleCardAdapter scheduleCardAdapter;
    private ArrayList<ScheduleCard>scheduleCardArrayList= new ArrayList<>();
    private RecyclerView scheduleRV;//스케쥴 카드뷰 생성
    ImageButton schedule_add;
    ImageView select_date;

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.calendar_1);

        MaterialDatePicker.Builder<Pair<Long,Long>> materialDateBuilder = MaterialDatePicker.Builder.dateRangePicker();
        materialDateBuilder.setTitleText("SELECT A DATE");
        Pair p= Pair.create(MaterialDatePicker.thisMonthInUtcMilliseconds(),
                MaterialDatePicker.todayInUtcMilliseconds());
        materialDateBuilder.setSelection(p);
        materialDateBuilder.setTheme(R.style.Theme_App);


        MaterialCalendarView calendar = findViewById(R.id.cv_calendar);//스케쥴 추가 버튼 가져옴

        calendar.setOnRangeSelectedListener(new OnRangeSelectedListener() {
            @Override
            public void onRangeSelected(@NonNull MaterialCalendarView widget, @NonNull List<CalendarDay> dates) {
                scheduleRV = (RecyclerView) findViewById(R.id.scheduleRV);//activitiy_map : bottomsheet layout
                scheduleRV.setHasFixedSize(true);
                String d="";
                for(int i=0; i<dates.size(); i++){
                    d=dates.get(i).getMonth()+"월 "+dates.get(i).getDay()+"일";
                    scheduleCardArrayList.add(new ScheduleCard(d));
                }
                Log.d("size",scheduleCardArrayList.size()+""+d);
                scheduleCardAdapter =new ScheduleCardAdapter(getBaseContext(),scheduleCardArrayList);//content가 문제인거 같은데..
                LinearLayoutManager linearLayoutManager = new LinearLayoutManager(getBaseContext(), LinearLayoutManager.VERTICAL, false);
                scheduleRV.setLayoutManager(linearLayoutManager);
                scheduleRV.setAdapter(scheduleCardAdapter);

            }
        });

    }
    //장소 카드 만듣
    public void makeMapSpaceCard(View mainview,List dates) {
        scheduleRV = (RecyclerView) mainview.findViewById(R.id.scheduleRV);//activitiy_map : bottomsheet layout
        scheduleRV.setHasFixedSize(true);
        for(int i=0; i<dates.size(); i++){
            String d=dates.get(i).toString();
            scheduleCardArrayList.add(new ScheduleCard(d));
        }

        Log.d("size",scheduleCardArrayList.size()+"");
        scheduleCardAdapter =new ScheduleCardAdapter(mainview.getContext(),scheduleCardArrayList);//content가 문제인거 같은데..
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(mainview.getContext(), LinearLayoutManager.VERTICAL, false);
        scheduleRV.setLayoutManager(linearLayoutManager);
        scheduleRV.setAdapter(scheduleCardAdapter);
    }

}