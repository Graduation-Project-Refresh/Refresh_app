package com.example.refresh_selection;

import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.DatePicker;
import android.widget.ImageButton;
import android.widget.ImageView;

import com.google.android.material.datepicker.MaterialDatePicker;
import com.google.android.material.datepicker.MaterialPickerOnPositiveButtonClickListener;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.util.Pair;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

public class Calendar_java extends AppCompatActivity {
    ScheduleCardAdapter scheduleCardAdapter;
    private ArrayList<ScheduleCard>scheduleCardArrayList= new ArrayList<>();
    private RecyclerView scheduleRV;//스케쥴 카드뷰 생성

    DatePicker cal_picker;
    ImageButton schedule_add;
    ImageView select_date;
    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.calendar_kotlin);

        MaterialDatePicker.Builder<Pair<Long,Long>> materialDateBuilder = MaterialDatePicker.Builder.dateRangePicker();
        materialDateBuilder.setTitleText("SELECT A DATE");
        Pair p= Pair.create(MaterialDatePicker.thisMonthInUtcMilliseconds(),
                MaterialDatePicker.todayInUtcMilliseconds());
        materialDateBuilder.setSelection(p);
        materialDateBuilder.setTheme(R.style.Theme_App);
//

        final MaterialDatePicker materialDatePicker = materialDateBuilder.build();
        select_date= findViewById(R.id.select_date);
        select_date.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                materialDatePicker.show(getSupportFragmentManager(),"MATERIAL_DATE_PICKER");
            }
        });

//        cal_picker=findViewById(R.id.cal_picker);//달력 picker 가져옴
        schedule_add=findViewById(R.id.addPlaceToSchedule);//스케쥴 추가 버튼 가져옴
        schedule_add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

            }
        });
        materialDatePicker.addOnPositiveButtonClickListener(new MaterialPickerOnPositiveButtonClickListener<Pair<Long,Long>>() {
            @Override
            public void onPositiveButtonClick(Pair<Long,Long> selection) {
                Long startDate= selection.first;
                Long endDate= selection.second;
//                Log.d("startDate",startDate.toString());
//                Log.d("endDate",endDate.toString());
                Date start=new Date(startDate);
                Date end = new Date(endDate);


                String day=start.getMonth()+"월"+start.getDate()+"일";
                int s=start.getDate();
//                Log.d("startDate",start.toString());
//                Log.d("endDate",end.toString());
                int len=0;
                try{

                    //두날짜 사이의 시간 차이(ms)를 하루 동안의 ms(24시*60분*60초*1000밀리초) 로 나눈다.
                    long diffDay = (end.getTime() - start.getTime()) / (24*60*60*1000);
                    System.out.println(diffDay+"일");
                    len= (int) diffDay;
                }catch(Exception e){
                    e.printStackTrace();
                }
                
                String strFormat = "yyyyMMdd";    //strStartDate 와 strEndDate 의 format
                Date d1 = null,d2=null;
                //SimpleDateFormat 을 이용하여 startDate와 endDate의 Date 객체를 생성한다.
                SimpleDateFormat sdf = new SimpleDateFormat(strFormat);
                String temp=sdf.format(start);
                try {
                    d1 = sdf.parse(temp);
                    temp=sdf.format(end);
                    d2 =sdf.parse(temp);
                    Log.d("d1 d2",d1.toString()+" "+d2.toString());
                } catch (ParseException e) {
                    e.printStackTrace();
                }

                Calendar c1= Calendar.getInstance();
                Calendar c2 = Calendar.getInstance();
                c1.setTime(d1);
                c2.setTime(d2);
                String dates[]=new String[len+1];
                int count=0;
//                SimpleDateFormat sd = new SimpleDateFormat("yyyy.MM.dd"); // String 날짜 형식을 정의 합니다.

                while(c1.compareTo(c2)!=1){//시작날짜가 작거나 같을때 출력한다.
                    if(count<len+1){
//                        c1.getTime();
                        String nowDayTimeStr = sdf.format(c1.getTime());
                        dates[count]=nowDayTimeStr;
                        c1.add(Calendar.DATE,1);
                        System.out.println(dates[count]);
                        count++;

                    }

                }
                
                //확인 버튼 눌렀을때? 리사이클 뷰에 카드뷰 나타나야함
                scheduleRV = (RecyclerView) findViewById(R.id.scheduleRV);
                scheduleRV.setHasFixedSize(true);

                for (int i=0; i<dates.length;i++){
                    ScheduleCard sc=new ScheduleCard(dates[i]);
                    scheduleCardArrayList.add(sc);
                }

                scheduleCardAdapter =new ScheduleCardAdapter(getApplicationContext(),scheduleCardArrayList);
                LinearLayoutManager linearLayoutManager = new LinearLayoutManager(getApplicationContext(), LinearLayoutManager.VERTICAL, false);
                scheduleRV.setLayoutManager(linearLayoutManager);
                scheduleRV.setAdapter(scheduleCardAdapter);
            }
        });
//
//        cal_picker.setOnDateChangedListener(new DatePicker.OnDateChangedListener() {
//            @Override
//            public void onDateChanged(DatePicker view, int year, int monthOfYear, int dayOfMonth) {
//
//            }
//        });

//        Calendar c = Calendar.getInstance();
//        int mYear = c.get(Calendar.YEAR);
//        int mMonth = c.get(Calendar.MONTH);
//        int mDay = c.get(Calendar.DAY_OF_MONTH);
//
//        DatePickerDialog datePickerDialog = new DatePickerDialog(this, new DatePickerDialog.OnDateSetListener() {
//            @Override
//            public void onDateSet(DatePicker view, int year, int month, int dayOfMonth) {
//                //date.setText(dayOfMonth+"/" + (month+1) + "/" + year);
//                Log.d("date",dayOfMonth+"/"+(month+1)+"/"+year);
//            }
//        }, mYear, mMonth, mDay);
//        datePickerDialog.show();





        //datePicker에 해당하는 프래그먼트 있어야한다.
//        DatePickerFragment DPF= new DatePickerFragment();
//        DPF.show(getSupportFragmentManager(),"MATERIAL_DATE_PICKER");


//        materialDatePicker.addOnPositiveButtonClickListener(
//                new MaterialPickerOnPositiveButtonClickListener() {
//                    @Override
//                    public void onPositiveButtonClick(Object selection) {
//                        Log.d("date",materialDatePicker.getHeaderText());
//                    }
//                });
//        materialDatePicker.addOnPositiveButtonClickListener(
//                new MaterialPickerOnPositiveButtonClickListener() {
//                    @Override
//                    public void onPositiveButtonClick(Object selection) {
//                        Log.d("date",materialDatePicker.getHeaderText());
//                    }
//                });

    }
    public void makeScheduleCard(ViewGroup mainview,String []dates) {
        scheduleRV = (RecyclerView) mainview.findViewById(R.id.scheduleRV);
        scheduleRV.setHasFixedSize(true);
        scheduleCardArrayList= new ArrayList<>();
        for (int i=0; i<dates.length;i++){
            scheduleCardArrayList.add(new ScheduleCard(dates[i]));
        }


        scheduleCardAdapter =new ScheduleCardAdapter(mainview.getContext(),scheduleCardArrayList);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(mainview.getContext(), LinearLayoutManager.VERTICAL, false);
        scheduleRV.setLayoutManager(linearLayoutManager);
        scheduleRV.setAdapter(scheduleCardAdapter);
    }

}
