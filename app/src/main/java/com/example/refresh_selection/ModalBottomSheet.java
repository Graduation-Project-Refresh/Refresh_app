package com.example.refresh_selection;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.android.material.bottomsheet.BottomSheetDialogFragment;

public class ModalBottomSheet extends BottomSheetDialogFragment {
    String TAG = "bottomSheet";
    private BottomSheetListener mListener;
    private View view;

    public ModalBottomSheet(){//empty constructor

    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
//        mListener = (BottomSheetListener) getContext();
        view = inflater.inflate(R.layout.bottom_sheet, container, false);
        return view;
    }

    public interface BottomSheetListener {
        void onButtonClicked();
    }
}
