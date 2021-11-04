package com.example.refresh_selection;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class ScheduleCardAdapter extends RecyclerView.Adapter<ScheduleCardAdapter.Viewholder> {
    private Context context;
    private ArrayList<ScheduleCard> ScheduleCardArrayList;

    public ScheduleCardAdapter(Context context, ArrayList<ScheduleCard> ScheduleCardArrayList){
        this.context=context;
        this.ScheduleCardArrayList=ScheduleCardArrayList;
    }

    @NonNull
    @Override
    public ScheduleCardAdapter.Viewholder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.schedule_cardview,parent,false);
        return new ScheduleCardAdapter.Viewholder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ScheduleCardAdapter.Viewholder holder, int position) {
        ScheduleCard sc=ScheduleCardArrayList.get(position);
        holder.schedule_date.setText(sc.getSchedule_date());
        //뷰는 뭐로 채워야할까?
    }

    @Override
    public int getItemCount() {
        return ScheduleCardArrayList.size();
    }

    public class Viewholder extends RecyclerView.ViewHolder {
        private TextView schedule_date;
        private View schedule_view;
        public Viewholder(@NonNull View itemView) {
            super(itemView);
//            schedule_date=itemView.findViewById();
        }
    }
}
