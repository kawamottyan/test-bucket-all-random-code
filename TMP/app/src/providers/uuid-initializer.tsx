'use client';

import { useEffect } from 'react';
import { v4 as uuidv4 } from "uuid";

export default function UuidInitializer() {
    useEffect(() => {
        let uuid = localStorage.getItem('uuid');

        if (!uuid) {
            uuid = uuidv4();
            localStorage.setItem('uuid', uuid);
        }
    }, []);

    return null;
}